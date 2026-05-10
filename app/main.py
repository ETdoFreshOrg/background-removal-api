import io
import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bg-remove")

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

log.info("All imports successful")

MODEL_ID = os.getenv("RMBG_MODEL", "briaai/RMBG-2.0")
API_KEY = os.getenv("API_KEY")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

_model = None
_transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _model
    try:
        threads = int(os.getenv("TORCH_THREADS", "4"))
        torch.set_num_threads(threads)
        log.info(f"Loading model {MODEL_ID} (CPU, {threads} threads)...")
        sys.stdout.flush()
        _model = (
            AutoModelForImageSegmentation.from_pretrained(
                MODEL_ID, trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
            )
            .eval()
        )
        log.info(f"Model loaded: {MODEL_ID}")
        sys.stdout.flush()
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    yield
    log.info("Shutting down")


app = FastAPI(title="Background Removal API", version="1.0.0", lifespan=lifespan)

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BG Remove</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f0f0f; color: #e0e0e0; min-height: 100vh; display: flex;
         flex-direction: column; align-items: center; padding: 2rem; }
  h1 { font-size: 1.8rem; margin-bottom: 0.3rem; }
  p.sub { color: #888; margin-bottom: 2rem; }
  #drop-zone {
    width: 100%; max-width: 600px; border: 2px dashed #444; border-radius: 16px;
    padding: 3rem 2rem; text-align: center; cursor: pointer; transition: all 0.2s;
    background: #1a1a1a; position: relative;
  }
  #drop-zone:hover, #drop-zone.drag-over { border-color: #646cff; background: #1e1e2e; }
  #drop-zone .icon { font-size: 3rem; margin-bottom: 0.5rem; }
  #drop-zone .label { color: #aaa; font-size: 0.95rem; }
  #drop-zone .label span { color: #646cff; text-decoration: underline; }
  #drop-zone .hint { color: #555; font-size: 0.8rem; margin-top: 0.5rem; }
  #file-input { display: none; }
  #status { margin-top: 1.5rem; min-height: 1.5rem; color: #888; }
  #status.error { color: #ff6b6b; }
  #paste-preview { display:none; margin-top:1rem; max-width:200px; max-height:200px; border-radius:8px; }
  #result-area {
    display: none; margin-top: 2rem; width: 100%; max-width: 600px;
    background: #1a1a1a; border-radius: 16px; overflow: hidden;
  }
  #result-area .header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 1rem 1.5rem; border-bottom: 1px solid #2a2a2a;
  }
  #result-area .header span { font-weight: 600; }
  #preview-wrap {
    padding: 1.5rem; display: flex; justify-content: center;
    background: repeating-conic-gradient(#2a2a2a 0% 25%, #222 0% 50%) 50%/20px 20px;
  }
  #preview { max-width: 100%; max-height: 450px; border-radius: 8px; }
  .btn {
    background: #646cff; color: #fff; border: none; padding: 0.55rem 1.2rem;
    border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600;
    transition: background 0.15s;
  }
  .btn:hover { background: #535bf2; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  #new-btn { margin-top: 1.5rem; }
</style>
</head>
<body>
<h1>BG Remove</h1>
<p class="sub">Drop an image or click to upload. Background removed instantly.</p>
<div id="drop-zone">
  <div class="icon">🖼️</div>
  <div class="label">Drag & drop an image here, or <span>browse</span></div>
  <div class="hint">You can also Ctrl+V to paste from clipboard</div>
  <input type="file" id="file-input" accept="image/png,image/jpeg,image/webp">
  <img id="paste-preview">
</div>
<div id="status"></div>
<div id="result-area">
  <div class="header">
    <span>Result</span>
    <div>
      <button class="btn" id="download-btn">Download</button>
    </div>
  </div>
  <div id="preview-wrap"><img id="preview"></div>
</div>
<button class="btn" id="new-btn" style="display:none">Remove another</button>

<script>
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const status = document.getElementById('status');
const resultArea = document.getElementById('result-area');
const preview = document.getElementById('preview');
const downloadBtn = document.getElementById('download-btn');
const newBtn = document.getElementById('new-btn');
let resultBlob = null;

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });
newBtn.addEventListener('click', resetUI);

document.addEventListener('paste', e => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      e.preventDefault();
      const file = item.getAsFile();
      if (file) {
        const named = new File([file], 'pasted-image.' + (file.type.split('/')[1] || 'png'), { type: file.type });
        handleFile(named);
      }
      return;
    }
  }
});

async function handleFile(file) {
  if (!file.type.startsWith('image/')) { setStatus('Please upload an image file.', true); return; }
  setStatus('Removing background...');
  dropZone.style.display = 'none';
  try {
    const form = new FormData();
    form.append('file', file);
    const resp = await fetch('/remove-bg', { method: 'POST', body: form });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(text || 'Server error ' + resp.status);
    }
    resultBlob = await resp.blob();
    preview.src = URL.createObjectURL(resultBlob);
    resultArea.style.display = 'block';
    downloadBtn.onclick = () => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(resultBlob);
      a.download = file.name.replace(/\\.[^.]+$/, '') + '-nobg.png';
      a.click();
    };
    newBtn.style.display = 'inline-block';
    setStatus('');
  } catch (err) {
    setStatus('Error: ' + err.message, true);
    dropZone.style.display = '';
  }
}

function setStatus(msg, isErr) {
  status.textContent = msg;
  status.className = isErr ? 'error' : '';
}

function resetUI() {
  resultArea.style.display = 'none';
  preview.src = '';
  resultBlob = null;
  fileInput.value = '';
  dropZone.style.display = '';
  newBtn.style.display = 'none';
  setStatus('');
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML


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
