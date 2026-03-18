import io
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="Handwritten to Text")

UI_FILE = PROJECT_ROOT / "ui" / "ui.html"
MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_WEIGHTS_PATH = MODEL_DIR / "model.pth"
FALLBACK_WEIGHTS_PATH = MODEL_DIR / "model_best.pth"
CHECKPOINT_PATH = MODEL_DIR / "last_checkpoint.pt"

CHAR_SET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?()-/&%$#@+*=<>[]{}"
char_to_idx = {c: i + 1 for i, c in enumerate(CHAR_SET)}
idx_to_char = {v: k for k, v in char_to_idx.items()}

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


class CRNN(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

        self.rnn = nn.LSTM(
            input_size=128 * 8,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )

        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.head_dropout(x)
        x = self.fc(x)
        return x.log_softmax(2)


def get_weights_path() -> Path:
    override = os.getenv("MODEL_WEIGHTS_PATH", "").strip()
    if override:
        return Path(override)
    if DEFAULT_WEIGHTS_PATH.exists():
        return DEFAULT_WEIGHTS_PATH
    if FALLBACK_WEIGHTS_PATH.exists():
        return FALLBACK_WEIGHTS_PATH
    return DEFAULT_WEIGHTS_PATH


def model_available() -> bool:
    return get_weights_path().exists()


def _extract_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def _load_model(weights_path: Path):
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. Put model.pth in model folder or set MODEL_WEIGHTS_PATH."
        )

    model = CRNN(num_classes=len(char_to_idx) + 1).to(device)
    payload = torch.load(weights_path, map_location=device)
    state_dict = _extract_state_dict(payload)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def decode(pred):
    pred = pred.argmax(2)[0]
    prev = -1
    text = ""

    for p in pred:
        idx = p.item()
        if idx != prev and idx != 0:
            text += idx_to_char.get(idx, "")
        prev = idx

    return text


def beam_search_decode(log_probs, beam_width=5):
    time_steps = log_probs.size(0)
    probs = log_probs.exp()
    beams = [("", -1, 1.0)]

    for t in range(time_steps):
        new_beams = []
        top_values, top_indices = probs[t, 0].topk(beam_width)
        for prefix, prev_token, score in beams:
            for v, idx in zip(top_values.tolist(), top_indices.tolist()):
                if idx == 0 or idx == prev_token:
                    new_prefix = prefix
                else:
                    new_prefix = prefix + idx_to_char.get(idx, "")
                new_beams.append((new_prefix, idx, score * v))

        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_width]

    return beams[0][0] if beams else ""


def predict(image: Image.Image, decode_method: str = "greedy"):
    global _model

    if _model is None:
        _model = _load_model(get_weights_path())

    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = _model(img)

    if decode_method == "beam":
        return beam_search_decode(pred.permute(1, 0, 2), beam_width=5)
    return decode(pred)

@app.get("/")
def home():
    if UI_FILE.exists():
        return FileResponse(UI_FILE)
    return {"message": "UI file not found. Open /docs to use API."}


@app.get("/ui")
def ui_page():
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail="UI file not found.")
    return FileResponse(UI_FILE)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/model-status")
def model_status():
    weights_path = get_weights_path()
    return {
        "ready": model_available(),
        "weights_path": str(weights_path),
        "device": str(device),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
    }

@app.post("/predict")
async def predict_text(file: UploadFile = File(...), decode_method: str = "greedy"):
    if decode_method not in {"greedy", "beam"}:
        raise HTTPException(status_code=400, detail="decode_method must be 'greedy' or 'beam'.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to parse image file.") from exc

    try:
        text = predict(image, decode_method=decode_method)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed unexpectedly.") from exc

    return {"prediction": text}