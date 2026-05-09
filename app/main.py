import io
import logging
import os
import traceback
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bg-remove")

try:
    from PIL import Image
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    log.info("All imports successful")
except Exception as e:
    log.error(f"Import failed: {e}")
    traceback.print_exc()
    raise

MODEL_ID = os.getenv("RMBG_MODEL", "briaai/RMBG-2.0")
API_KEY = os.getenv("API_KEY")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

app = FastAPI(title="Background Removal API", version="1.0.0")

_model = None
_transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@app.on_event("startup")
def load_model():
    global _model
    try:
        threads = int(os.getenv("TORCH_THREADS", "4"))
        torch.set_num_threads(threads)
        log.info(f"Loading model {MODEL_ID} (CPU, {threads} threads)...")
        _model = (
            AutoModelForImageSegmentation.from_pretrained(
                MODEL_ID, trust_remote_code=True
            )
            .eval()
        )
        log.info(f"Model loaded: {MODEL_ID}")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        traceback.print_exc()
        raise


def verify_api_key(authorization: Optional[str] = Header(None)):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.split(" ", 1)[1].strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    loaded = _model is not None
    return {
        "status": "ok" if loaded else "loading",
        "model": MODEL_ID,
        "device": "cpu",
    }


@app.post("/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_MB}MB)",
        )

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    input_tensor = _transform(image).unsqueeze(0)
    with torch.no_grad():
        preds = _model(input_tensor)[-1].sigmoid().cpu()
    mask = transforms.ToPILImage()(preds[0].squeeze()).resize(image.size)
    image.putalpha(mask)

    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return Response(content=buf.getvalue(), media_type="image/png")
