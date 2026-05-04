import os
import io
import base64
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

from typing import List

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = "../best_model.keras"
best_model = tf.keras.models.load_model(MODEL_PATH)

# ── Class mapping ─────────────────────────────────────────────────────────────
CLASS_INDEX_TO_LABEL: dict[int, str] = {
    0: "Ayam Goreng",
    1: "Burger",
    2: "French Fries",
    3: "Gado-Gado",
    4: "Ikan Goreng",
    5: "Mie Goreng",
    6: "Nasi Goreng",
    7: "Nasi Padang",
    8: "Pizza",
    9: "Rawon",
    10: "Rendang",
    11: "Sate",
    12: "Soto",
}

# ── Price map ─────────────────────────────────────────────────────────────────
PRICE_MAP: dict[str, int] = {
    "Ayam Goreng": 10_000,
    "Burger":      15_000,
    "French Fries":12_000,
    "Gado-Gado":   10_000,
    "Ikan Goreng":  8_000,
    "Mie Goreng":  10_000,
    "Nasi Goreng": 13_000,
    "Nasi Padang": 15_000,
    "Pizza":       20_000,
    "Rawon":       12_000,
    "Rendang":     15_000,
    "Sate":        14_000,
    "Soto":        12_000,
}

# ─── Model inferensi ────────────────────────────────────────────────────────
def run_model(img_array: np.ndarray) -> tuple[str, float]:
    """Jalankan inferensi model. Return (label, confidence 0–1)."""
    processed = preprocess_input(np.expand_dims(img_array, axis=0))
    preds     = best_model.predict(processed, verbose=0)
    idx       = int(np.argmax(preds[0]))
    score     = float(np.max(preds[0]))
    label     = CLASS_INDEX_TO_LABEL[idx]   
    return label, score


# ── FastAPI setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Kantin Digital")

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static",
)

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        if not file.content_type.startswith("image/"): continue
        raw = await file.read()
        try:
            pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
            img_array = np.array(pil_img.resize((224, 224)), dtype=np.float32)
            label, confidence = run_model(img_array)
            
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=70)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            results.append({
                "label": label,
                "confidence": round(confidence * 100, 2),
                "price": PRICE_MAP.get(label, 0),
                "image_b64": img_b64
            })
        except: continue
    
    if not results: raise HTTPException(status_code=400, detail="Gagal memproses gambar.")
    return results


@app.get("/menu")
async def menu():
    """Kembalikan daftar harga menu."""
    items = [
        {"name": k, "price": v}
        for k, v in sorted(PRICE_MAP.items(), key=lambda x: x[0])
    ]
    return JSONResponse({"items": items})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)