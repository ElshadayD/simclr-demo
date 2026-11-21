# Lumen: chest X-ray screening app

1. `pip install -r requirements.txt`
2. `python app.py` then open http://127.0.0.1:5000

The model comes from Hugging Face (`Elshaday/SimCLR_medical_image_classification`).
It downloads once into `.model_cache/` on the first run; every later start loads it from disk in seconds.
For a demo, start the app a few minutes beforehand so it is already warm.

Optional environment variables:
- `HF_TOKEN`: access token, needed only if the repo is private
- `HF_REPO` / `HF_FILE`: use a different repo or file name
- `MODEL_PATH`: load a local .pt file instead of Hugging Face

Research demo only, not for clinical use.
