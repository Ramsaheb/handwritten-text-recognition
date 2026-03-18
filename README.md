# Handwritten Text Recognition (Base Prototype)

A FastAPI + HTML web app that predicts handwritten text from uploaded images using a CRNN model.

This is a base prototype focused on word-level recognition and end-to-end deployment workflow.

## Highlights
- FastAPI backend for inference API.
- Clean web UI for upload and prediction.
- Greedy and beam decoding options.
- Runtime model status endpoint.
- Multi-line input warning behavior for unsupported cases.

## Interview Outputs

Use these file paths for your interview screenshots:

- `docs/screenshots/interview-output-dear.png`
- `docs/screenshots/interview-output-good.png`

After adding those files, these previews will render automatically:

### Interview Output 1 (Dear)
![Interview Output Dear](docs/screenshots/interview-output-dear.png)

### Interview Output 2 (Good)
![Interview Output Good](docs/screenshots/interview-output-good.png)

## What This Prototype Can Do
- Predict text for single handwritten word images.
- Load a trained model checkpoint from the `model` folder.
- Return prediction and warning messages through API and UI.
- Provide model readiness info through `/model-status`.

## Current Limitations
- Not a full paragraph OCR system.
- Trained mainly on IAM word-level samples.
- Multi-line sentence images can return partial/noisy output.
- No language-model post-correction.

## Recommended Input for Best Results
- One tightly cropped handwritten word.
- High contrast and clean background.
- Minimal blur/noise.

## Dataset
- IAM dataset: https://www.kaggle.com/datasets/ngkinwang/iam-dataset

## Tech Stack
- Python
- FastAPI
- PyTorch
- Torchvision
- Pillow
- NumPy

## Project Structure
- `app/main.py`: FastAPI app, model loading, preprocessing, decoding, inference.
- `ui/ui.html`: frontend upload and prediction UI.
- `model/`: runtime model checkpoints (`.pth`, `.pt`).
- `train_model.ipynb`: training and evaluation notebook.
- `requirements.txt`: runtime dependencies.

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

3. Put checkpoint file in the `model` folder.

## Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

## Model Path Behavior
- Primary path: `model/model.pth`
- Fallback path: `model/model_best.pth`
- Optional override using environment variable:

```bash
MODEL_WEIGHTS_PATH=/path/to/weights.pth
```

## Notes
- Large data and model artifacts are ignored using `.gitignore`.
- `train_model.ipynb` is kept for retraining/future improvements.
- This repository is a base version; production OCR needs stronger segmentation and line-level modeling.
