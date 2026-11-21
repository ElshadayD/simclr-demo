"""Lumen: chest X-ray pneumonia screening (SimCLR ResNet50, fine-tuned layer4 + linear head).

Works locally (`python app.py`) and on Render (`gunicorn app:app`, see render.yaml).
"""
import base64, ctypes, gc, io, os, resource, threading, traceback, urllib.request

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from werkzeug.exceptions import HTTPException

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HF_REPO = os.environ.get("HF_REPO", "Elshaday/SimCLR_medical_image_classification")
HF_FILE = os.environ.get("HF_FILE", "linear_eval_simclr_cxr.pt")
LOCAL_PATH = os.environ.get("MODEL_PATH")  # optional: use a local file instead of Hugging Face
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", os.path.join(BASE_DIR, ".model_cache"))
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # ImageFolder order used in training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Render instances have few CPU cores; more threads than cores just slows things down.
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))


def rss_mb():
    """Current resident memory of this process in MB (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024)
    except Exception:
        pass
    return None


def peak_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)  # KB on Linux


def release_memory():
    """Hand freed heap memory back to the OS so RSS drops after each request."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class PartialFineTuneModel(nn.Module):
    """Same structure as the training notebook so the state_dict keys match."""
    def __init__(self, num_classes=2):
        super().__init__()
        b = resnet50(weights=None)
        b.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        b.maxpool = nn.Identity()
        b.fc = nn.Identity()
        self.backbone = b
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))


state = {"ready": False, "error": None, "stage": "Preparing the model", "progress": None}
model = None
_infer_lock = threading.Lock()  # one inference at a time keeps memory use predictable


def load_model():
    if LOCAL_PATH:
        src = LOCAL_PATH
    else:
        cache = os.path.join(CACHE_DIR, HF_FILE)
        if not os.path.exists(cache):  # download once per container start
            os.makedirs(CACHE_DIR, exist_ok=True)
            url = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_FILE}"
            req = urllib.request.Request(url)
            if os.environ.get("HF_TOKEN"):  # only needed for private repos
                req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
            try:
                with urllib.request.urlopen(req, timeout=120) as r, open(cache + ".part", "wb") as out:
                    total = int(r.headers.get("Content-Length") or 0)
                    done = 0
                    while chunk := r.read(1 << 20):
                        out.write(chunk)
                        done += len(chunk)
                        if total:
                            state["progress"] = round(100 * done / total)
                os.replace(cache + ".part", cache)
            except Exception as e:
                raise RuntimeError(f"Could not fetch the model from Hugging Face: {e}. "
                                   "If the repo is private, set HF_TOKEN. If offline, set MODEL_PATH.")
        src = cache
    state["stage"] = "Loading the model"
    m = PartialFineTuneModel(len(CLASS_NAMES))
    try:  # mmap avoids holding a second full copy of the weights in RAM while loading
        sd = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        try:
            sd = torch.load(src, map_location="cpu", weights_only=True)
        except Exception:  # checkpoint holds non-tensor objects: use the full loader
            sd = torch.load(src, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd)
    del sd
    release_memory()
    m = m.to(device).eval()
    for p in m.parameters():  # inference only: no parameter gradients needed
        p.requires_grad_(False)
    return m


tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def boot():
    """Runs in the background so the web server can answer health checks immediately."""
    global model
    try:
        m = load_model()
        state["stage"] = "Warming up"
        with torch.no_grad():  # one dummy pass so the first real scan is fast
            m(torch.zeros(1, 3, 224, 224, device=device))
        model = m
        release_memory()
        state["ready"] = True
    except Exception as e:
        traceback.print_exc()
        state["error"] = str(e)


def colorize(cam):
    """Map 0..1 array to a blue->green->yellow->red heatmap (no matplotlib needed)."""
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def to_b64(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze(img, tta=False):
    """Prediction + Grad-CAM on the last conv block (layer4).

    Layers before layer4 run without autograd; only layer4 + head build a graph.
    Same maths as hooking layer4, but cheaper in memory and with no shared global state.
    """
    x = tf(img).unsqueeze(0).to(device)
    b = model.backbone
    with _infer_lock:
        with torch.no_grad():
            h = b.maxpool(b.relu(b.bn1(b.conv1(x))))
            h = b.layer3(b.layer2(b.layer1(h)))
        h = h.requires_grad_(True)
        with torch.enable_grad():
            act = b.layer4(h)                                   # (1, 2048, H, W)
            logits = model.classifier(act.mean((2, 3)))         # avgpool + linear head
            probs = torch.softmax(logits, 1)[0]
            cls = int(probs.argmax())
            grad = torch.autograd.grad(logits[0, cls], act)[0]
        w = grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * act.detach()).sum(1, keepdim=True))
        cam = F.interpolate(cam, (224, 224), mode="bilinear", align_corners=False)[0, 0]
        cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).cpu().numpy()
        probs = probs.detach()
        if tta:  # average with horizontally flipped view for a steadier estimate
            with torch.no_grad():
                pf = torch.softmax(model(torch.flip(x, [3])), 1)[0]
            probs = (probs + pf) / 2
        probs = probs.cpu().numpy()

    base = img.convert("RGB").resize((224, 224))
    heat = Image.fromarray(colorize(cam))
    return {
        "probs": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        "heatmap": to_b64(heat),
        "overlay": to_b64(Image.blend(base, heat, 0.45)),
    }


app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, HTTPException):  # 404, 413, ... keep their real status code
        return jsonify(error=e.description), e.code
    traceback.print_exc()
    return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.get("/")
def index():
    return render_template("index.html", device=device)


@app.get("/healthz")
def healthz():
    """Render health check: returns 200 as soon as the server is up (model may still be loading)."""
    return jsonify(ok=True, ready=state["ready"])


@app.get("/api/status")
def status():
    return jsonify({**state, "rss_mb": rss_mb(), "peak_mb": peak_mb()})


@app.post("/api/predict")
def predict():
    if not state["ready"]:
        return jsonify(error="The model is still starting up."), 503
    tta = request.form.get("tta") == "1"
    out = []
    for f in request.files.getlist("images"):
        try:
            img = Image.open(f.stream)
            img.draft("RGB", (2048, 2048))  # JPEG only: decode at reduced size
            img = img.convert("RGB")
            if max(img.size) > 2048:        # huge scans: shrink before the 224x224 resize
                img.thumbnail((2048, 2048))
            res = analyze(img, tta)
            res["name"] = f.filename
            out.append(res)
            release_memory()
        except Exception as e:
            traceback.print_exc()
            out.append({"name": f.filename, "error": f"Analysis failed: {e}"})
    return jsonify(out)


# Start loading the model when the module is imported, so it also happens under gunicorn
# (which never runs the __main__ block below).
_boot_started = False
def start_boot():
    global _boot_started
    if not _boot_started:
        _boot_started = True
        threading.Thread(target=boot, daemon=True).start()

start_boot()


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"Open http://{host}:{port}  (the model loads in the background)")
    app.run(host=host, port=port, debug=False)
