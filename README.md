# Dataset
https://www.kaggle.com/datasets/ngkinwang/iam-dataset

# Handwritten Text Images to Text Conversion Web App (Base Version)

This is a base FastAPI + HTML UI project for handwritten text recognition using a CRNN model.

## Base Version: What It Can Do
- Serve a web UI for image upload and prediction.
- Run inference from a `.pth` checkpoint in the `model/` folder.
- Support decode modes: `greedy` and `beam`.
- Return prediction output through API and UI.
- Provide model status through `/model-status`.
- Warn users when input looks like multi-line text or low-confidence output.

## Base Version: What It Cannot Do (Yet)
- It is not a full paragraph OCR model.
- It is mainly trained for IAM-style word images (single-word crops).
- Multi-line sentence images may produce partial or noisy text.
- It does not include robust document text detection/segmentation like production OCR engines.
- It does not auto-correct text using a language model.

## Recommended Input
- Best: one handwritten word cropped tightly.
- Good contrast between text and background.
- Avoid large page images if you need accurate sentence-level transcription.

## Project Structure
- `app/main.py`: FastAPI app, model loading, decoding, and inference API.
- `ui/ui.html`: frontend upload and prediction UI.
- `model/`: model weights/checkpoints for runtime.
- `requirements.txt`: runtime dependencies.
- `train_model.ipynb`: training and evaluation notebook.

## API Endpoints
- `GET /`: serve UI page.
- `GET /ui`: serve UI page.
- `GET /model-status`: model readiness, path, runtime info.
- `POST /predict`: image prediction endpoint.

## Setup
1. Activate your Python/Conda environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put weights in `model/model.pth` (or `model/model_best.pth`).

## Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

## Model Weights
- Default path: `model/model.pth`
- Fallback path: `model/model_best.pth`
- Optional override:

```bash
MODEL_WEIGHTS_PATH=/path/to/weights.pth
```

## Notes
- Large data and model artifacts are ignored via `.gitignore`.
- `train_model.ipynb` is kept for retraining and future improvements.
- This repository is a base version; improve segmentation/modeling for production-level sentence OCR.
