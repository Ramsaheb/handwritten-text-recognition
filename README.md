# Dataset
https://www.kaggle.com/datasets/ngkinwang/iam-dataset

# Handwritten Text Images to Text Conversion Web App

FastAPI backend + simple HTML UI for handwritten word image recognition using a CRNN model.

## What Is Kept
- Backend API: app/main.py
- Inference runtime: integrated directly in app/main.py
- UI page: ui/ui.html
- Training notebook (kept as requested): train_model.ipynb

## Project Structure
- app/main.py: FastAPI app and endpoints
- model/: stores weights/checkpoints only (for runtime)
- ui/ui.html: frontend UI
- requirements.txt: runtime dependencies
- train_model.ipynb: training workflow notebook

## API Endpoints
- GET /: serves UI page
- GET /ui: serves UI page
- GET /model-status: model availability and weight path
- POST /predict: image prediction endpoint

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   pip install -r requirements.txt
3. Ensure model weights exist at model/model.pth or set MODEL_WEIGHTS_PATH.

## Run
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Open:
- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

## Model Weights
- Default location: model/model.pth
- Fallback location: model/model_best.pth
- Override with environment variable:
  MODEL_WEIGHTS_PATH=/path/to/weights.pth

## Notes
- Large dataset and model artifacts are ignored by .gitignore.
- Use train_model.ipynb for training and export final weights for runtime inference.
