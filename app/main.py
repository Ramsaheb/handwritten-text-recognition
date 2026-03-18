import io
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
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


def _preprocess_for_model(image: Image.Image) -> Image.Image:
    # Keep preprocessing conservative so it improves noisy inputs without drifting from training distribution.
    img = image.convert("L")
    img = ImageOps.autocontrast(img, cutoff=2)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
    img = ImageEnhance.Contrast(img).enhance(1.12)

    arr = np.array(img)
    ink_mask = arr < 235
    ys, xs = np.where(ink_mask)
    if ys.size > 0 and xs.size > 0:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        h, w = arr.shape
        pad_y = max(3, int(0.03 * h))
        pad_x = max(3, int(0.03 * w))
        y0 = max(0, y0 - pad_y)
        y1 = min(h - 1, y1 + pad_y)
        x0 = max(0, x0 - pad_x)
        x1 = min(w - 1, x1 + pad_x)
        img = img.crop((x0, y0, x1 + 1, y1 + 1))

    border = max(4, int(0.04 * max(img.size)))
    img = ImageOps.expand(img, border=border, fill=255)
    return img


def predict(image: Image.Image, decode_method: str = "greedy"):
    global _model

    if _model is None:
        _model = _load_model(get_weights_path())

    processed = _preprocess_for_model(image)
    img = transform(processed).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = _model(img)

    if decode_method == "beam":
        return beam_search_decode(pred.permute(1, 0, 2), beam_width=5)
    return decode(pred)


def _find_segments_1d(signal, min_value, min_len):
    segments = []
    start = None
    for i, v in enumerate(signal):
        if v >= min_value and start is None:
            start = i
        elif v < min_value and start is not None:
            if i - start >= min_len:
                segments.append((start, i - 1))
            start = None
    if start is not None and len(signal) - start >= min_len:
        segments.append((start, len(signal) - 1))
    return segments


def _merge_close_segments(segments, max_gap):
    if not segments:
        return []
    merged = [segments[0]]
    for s, e in segments[1:]:
        ps, pe = merged[-1]
        if s - pe <= max_gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _segment_word_crops(image: Image.Image):
    gray = image.convert("L")
    arr = np.array(gray)
    h, w = arr.shape

    # Simple adaptive threshold to estimate ink mask.
    threshold = int(min(220, max(80, arr.mean() * 0.9)))
    ink = arr < threshold

    if ink.sum() < max(50, int(0.0008 * h * w)):
        return []

    row_signal = ink.sum(axis=1)
    row_min = max(2, int(0.02 * w))
    row_segments = _find_segments_1d(row_signal, row_min, min_len=6)
    row_segments = _merge_close_segments(row_segments, max_gap=6)

    crops = []
    for r0, r1 in row_segments:
        r0p = max(0, r0 - 4)
        r1p = min(h - 1, r1 + 4)
        line_ink = ink[r0p:r1p + 1, :]

        col_signal = line_ink.sum(axis=0)
        col_min = max(1, int(0.12 * (r1p - r0p + 1)))
        col_segments = _find_segments_1d(col_signal, col_min, min_len=8)
        col_segments = _merge_close_segments(col_segments, max_gap=max(8, int(0.015 * w)))

        for c0, c1 in col_segments:
            c0p = max(0, c0 - 3)
            c1p = min(w - 1, c1 + 3)
            if (c1p - c0p + 1) < 12 or (r1p - r0p + 1) < 10:
                continue
            crops.append(gray.crop((c0p, r0p, c1p + 1, r1p + 1)))

    return crops


def _looks_like_multiline_input(image: Image.Image) -> bool:
    gray = image.convert("L")
    arr = np.array(gray)
    h, w = arr.shape
    if h < 48 or w < 96:
        return False

    # Many sentence/note images are large compared to IAM word crops.
    if h >= 160 and w >= 320:
        return True

    threshold = int(min(220, max(80, arr.mean() * 0.9)))
    ink = arr < threshold
    row_signal = ink.sum(axis=1)
    row_min = max(2, int(0.02 * w))
    row_segments = _find_segments_1d(row_signal, row_min, min_len=6)
    row_segments = _merge_close_segments(row_segments, max_gap=6)
    return len(row_segments) >= 2


def predict_with_segmentation(image: Image.Image, decode_method: str = "greedy"):
    word_crops = _segment_word_crops(image)
    if not word_crops:
        return "", 0

    words = []
    for crop in word_crops:
        txt = predict(crop, decode_method=decode_method).strip()
        if txt:
            words.append(txt)

    return " ".join(words), len(word_crops)


def _looks_like_low_confidence_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    non_space = [c for c in stripped if not c.isspace()]
    if not non_space:
        return True

    alpha_count = sum(c.isalpha() for c in non_space)
    punct_count = sum((not c.isalnum()) for c in non_space)
    alpha_ratio = alpha_count / len(non_space)
    punct_ratio = punct_count / len(non_space)

    # Heuristic gate: mostly punctuation/symbols usually indicates bad OCR for this model.
    if alpha_ratio < 0.45 or punct_ratio > 0.35:
        return True

    return False

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
        multiline_input = _looks_like_multiline_input(image)
        text = predict(image, decode_method=decode_method)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed unexpectedly.") from exc

    segmented_used = False
    segments_found = 0
    if not text.strip():
        segmented_text, segments_found = predict_with_segmentation(image, decode_method=decode_method)
        if segmented_text.strip():
            text = segmented_text
            segmented_used = True

    warning = None
    if segmented_used and _looks_like_low_confidence_text(text):
        text = ""
        warning = (
            "Detected multi-line text, but output confidence is low for this model. "
            "This model is trained on single-word crops. Please upload a tight crop of one word."
        )
    elif multiline_input and len(text.split()) <= 1:
        warning = (
            "Detected multi-line text. This model is trained on single-word crops, so output may be incomplete. "
            "For better results, upload one word at a time."
        )
    elif not text.strip():
        warning = (
            "No text detected from this image. This model is trained for IAM word images; "
            "for best results, upload a tight single-word crop with clear contrast."
        )

    return {
        "prediction": text,
        "warning": warning,
        "decode_method": decode_method,
        "segmentation_used": segmented_used,
        "segments_found": segments_found,
    }