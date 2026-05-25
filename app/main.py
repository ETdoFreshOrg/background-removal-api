import io
import json
import logging
import os
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bg-remove")

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

log.info("All imports successful")

MODEL_ID = os.getenv("RMBG_MODEL", "briaai/RMBG-2.0")
API_KEY = os.getenv("API_KEY")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
HISTORY_DIR = Path(os.getenv("HISTORY_DIR", "/app/history"))
HISTORY_JSON = HISTORY_DIR / "history.json"
MAX_HISTORY = 100

_model = None
_device = torch.device("cpu")
_transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_history():
    if HISTORY_JSON.exists():
        try:
            return json.loads(HISTORY_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_history(entries):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_JSON.write_text(json.dumps(entries, indent=2))


def prune_history(entries):
    if len(entries) <= MAX_HISTORY:
        return entries
    to_remove = entries[: len(entries) - MAX_HISTORY]
    for entry in to_remove:
        jid = entry.get("id", "")
        for f in HISTORY_DIR.glob(f"{jid}_*"):
            try:
                f.unlink()
            except OSError:
                pass
    return entries[len(entries) - MAX_HISTORY :]


def add_history_entry(job_id, filename, processing_time_ms=None):
    entries = load_history()
    entries.append(
        {
            "id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "processingTimeMs": processing_time_ms,
        }
    )
    entries = prune_history(entries)
    save_history(entries)


def complete_history_entry(job_id, processing_time_ms):
    entries = load_history()
    for entry in entries:
        if entry.get("id") == job_id:
            entry["processingTimeMs"] = processing_time_ms
            break
    save_history(entries)


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _model, _device
    try:
        threads = int(os.getenv("TORCH_THREADS", "4"))
        torch.set_num_threads(threads)
        requested_device = os.getenv("TORCH_DEVICE", "auto").lower().strip()
        if requested_device == "auto":
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif requested_device == "cuda" and not torch.cuda.is_available():
            log.warning("TORCH_DEVICE=cuda requested, but CUDA is not available; falling back to CPU")
            _device = torch.device("cpu")
        else:
            _device = torch.device(requested_device)
        log.info(
            f"Loading model {MODEL_ID} on {_device} "
            f"(cuda_available={torch.cuda.is_available()}, {threads} CPU threads)..."
        )
        sys.stdout.flush()
        _model = (
            AutoModelForImageSegmentation.from_pretrained(
                MODEL_ID, trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
            )
            .eval()
            .to(_device)
        )
        log.info(f"Model loaded: {MODEL_ID}")
        sys.stdout.flush()
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    yield
    log.info("Shutting down")


app = FastAPI(title="Background Removal API", version="1.0.0", lifespan=lifespan)

# ─── Main UI ────────────────────────────────────────────────────────────────

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

  /* Drop zone */
  #drop-zone {
    width: 100%; max-width: 600px; border: 2px dashed #444; border-radius: 16px;
    padding: 3rem 2rem; text-align: center; cursor: pointer; transition: all 0.2s;
    background: #1a1a1a;
  }
  #drop-zone:hover, #drop-zone.drag-over { border-color: #646cff; background: #1e1e2e; }
  #drop-zone .icon { font-size: 3rem; margin-bottom: 0.5rem; }
  #drop-zone .label { color: #aaa; font-size: 0.95rem; }
  #drop-zone .label span { color: #646cff; text-decoration: underline; }
  #drop-zone .hint { color: #555; font-size: 0.8rem; margin-top: 0.5rem; }
  #file-input { display: none; }

  /* Processing area */
  #processing-area {
    display: none; width: 100%; max-width: 700px; text-align: center;
  }
  #orig-preview-wrap {
    background: #1a1a1a; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;
  }
  #orig-preview-wrap .file-info { color: #888; font-size: 0.85rem; margin-bottom: 0.5rem; }
  #orig-preview { max-width: 100%; max-height: 300px; border-radius: 8px; }

  /* Spinner */
  .spinner-wrap { display: flex; flex-direction: column; align-items: center; gap: 1rem; }
  .spinner {
    width: 48px; height: 48px; border: 4px solid #2a2a2a; border-top-color: #646cff;
    border-radius: 50%; animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner-label { color: #888; font-size: 0.95rem; }
  .spinner-label .elapsed { color: #646cff; font-weight: 600; }

  /* Progress bar */
  .progress-track {
    width: 100%; max-width: 400px; height: 6px; background: #2a2a2a;
    border-radius: 3px; overflow: hidden; margin-top: 0.5rem;
  }
  .progress-fill {
    height: 100%; background: #646cff; border-radius: 3px;
    transition: width 0.3s ease; width: 0%;
  }

  /* Result area — side by side */
  #result-area {
    display: none; width: 100%; max-width: 800px;
  }
  .compare-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
  }
  @media (max-width: 600px) { .compare-grid { grid-template-columns: 1fr; } }
  .compare-card {
    background: #1a1a1a; border-radius: 12px; overflow: hidden;
  }
  .compare-card .card-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.75rem 1rem; border-bottom: 1px solid #2a2a2a;
  }
  .compare-card .card-header span { font-weight: 600; font-size: 0.9rem; }
  .compare-card .time-badge {
    color: #646cff; font-size: 0.8rem; font-weight: 600;
  }
  .compare-card .img-wrap {
    padding: 1rem; display: flex; justify-content: center; align-items: center;
    min-height: 200px;
  }
  .compare-card .img-wrap.checker {
    background: repeating-conic-gradient(#2a2a2a 0% 25%, #222 0% 50%) 50%/20px 20px;
  }
  .compare-card img { max-width: 100%; max-height: 400px; border-radius: 6px; cursor: pointer; transition: transform 0.15s; }
  .compare-card img:hover { transform: scale(1.02); }
  .result-actions {
    display: flex; gap: 0.75rem; justify-content: center; margin-top: 1.5rem;
  }

  /* Buttons */
  .btn {
    background: #646cff; color: #fff; border: none; padding: 0.55rem 1.2rem;
    border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600;
    transition: background 0.15s;
  }
  .btn:hover { background: #535bf2; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn.secondary { background: #333; }
  .btn.secondary:hover { background: #444; }

  /* Lightbox */
  #lightbox {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.88);
    z-index: 100; flex-direction: column; align-items: center; justify-content: center;
  }
  #lightbox.active { display: flex; }
  #lb-close {
    position: absolute; top: 1rem; right: 1.5rem; font-size: 2rem;
    color: #aaa; cursor: pointer; background: none; border: none; z-index: 101;
  }
  #lb-close:hover { color: #fff; }
  #lb-img-wrap {
    display: flex; justify-content: center; align-items: center;
    max-width: 90vw; max-height: calc(90vh - 60px); border-radius: 8px;
    overflow: hidden; transition: background 0.2s;
  }
  #lb-img-wrap img { max-width: 90vw; max-height: calc(90vh - 60px); object-fit: contain; }
  #bg-bar {
    display: flex; gap: 6px; padding: 10px 16px; background: rgba(30,30,30,0.95);
    border-radius: 10px; margin-top: 12px; align-items: center;
  }
  #bg-bar .bar-label { color: #888; font-size: 0.75rem; margin-right: 4px; }
  .bg-swatch {
    width: 26px; height: 26px; border-radius: 6px; cursor: pointer; border: 2px solid transparent;
    transition: border-color 0.15s, transform 0.1s;
  }
  .bg-swatch:hover { transform: scale(1.15); }
  .bg-swatch.active { border-color: #646cff; }
  .bg-swatch.sw-checker-light {
    background: repeating-conic-gradient(#ddd 0% 25%, #fff 0% 50%) 50%/10px 10px;
  }
  .bg-swatch.sw-checker-dark {
    background: repeating-conic-gradient(#2a2a2a 0% 25%, #222 0% 50%) 50%/10px 10px;
  }

  #status { margin-top: 1.5rem; min-height: 1.5rem; color: #888; }
  #status.error { color: #ff6b6b; }
</style>
</head>
<body>
<h1>BG Remove</h1>
<p class="sub">Drop an image or click to upload. Background removed instantly.</p>

<div id="drop-zone">
  <div class="icon">&#x1F5BC;&#xFE0F;</div>
  <div class="label">Drag & drop an image here, or <span>browse</span></div>
  <div class="hint">You can also Ctrl+V to paste from clipboard</div>
  <input type="file" id="file-input" accept="image/png,image/jpeg,image/webp">
</div>

<div id="processing-area">
  <div id="orig-preview-wrap">
    <div class="file-info" id="file-info"></div>
    <img id="orig-preview">
  </div>
  <div class="spinner-wrap">
    <div class="spinner"></div>
    <div class="spinner-label">Removing background&hellip; <span class="elapsed" id="elapsed">0.0s</span></div>
    <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>
  </div>
</div>

<div id="result-area">
  <div class="compare-grid">
    <div class="compare-card">
      <div class="card-header"><span>Original</span></div>
      <div class="img-wrap"><img id="result-orig"></div>
    </div>
    <div class="compare-card">
      <div class="card-header">
        <span>Result</span>
        <span class="time-badge" id="time-badge"></span>
      </div>
      <div class="img-wrap checker"><img id="result-img"></div>
    </div>
  </div>
  <div class="result-actions">
    <button class="btn" id="download-btn">Download</button>
    <button class="btn secondary" id="new-btn">Remove another</button>
  </div>
</div>

<div id="lightbox">
  <button id="lb-close">&times;</button>
  <div id="lb-img-wrap"><img id="lb-img"></div>
  <div id="bg-bar" style="display:none">
    <span class="bar-label">BG:</span>
    <div class="bg-swatch active sw-checker-dark"  data-bg="checker-dark"  title="Dark checker"></div>
    <div class="bg-swatch sw-checker-light"         data-bg="checker-light" title="Light checker"></div>
    <div class="bg-swatch" style="background:#fff"  data-bg="#ffffff"       title="White"></div>
    <div class="bg-swatch" style="background:#aaa"  data-bg="#aaaaaa"       title="Grey"></div>
    <div class="bg-swatch" style="background:#000"  data-bg="#000000"       title="Black"></div>
    <div class="bg-swatch" style="background:#e74c3c" data-bg="#e74c3c"    title="Red"></div>
    <div class="bg-swatch" style="background:#2ecc71" data-bg="#2ecc71"    title="Green"></div>
    <div class="bg-swatch" style="background:#3498db" data-bg="#3498db"    title="Blue"></div>
    <div class="bg-swatch" style="background:#1abc9c" data-bg="#1abc9c"    title="Cyan"></div>
    <div class="bg-swatch" style="background:#f1c40f" data-bg="#f1c40f"    title="Yellow"></div>
    <div class="bg-swatch" style="background:#e84393" data-bg="#e84393"    title="Magenta"></div>
  </div>
</div>

<div id="status"></div>

<script>
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const processingArea = document.getElementById('processing-area');
const origPreview = document.getElementById('orig-preview');
const fileInfo = document.getElementById('file-info');
const elapsedEl = document.getElementById('elapsed');
const progressFill = document.getElementById('progress-fill');
const resultArea = document.getElementById('result-area');
const resultOrig = document.getElementById('result-orig');
const resultImg = document.getElementById('result-img');
const timeBadge = document.getElementById('time-badge');
const downloadBtn = document.getElementById('download-btn');
const newBtn = document.getElementById('new-btn');
const status = document.getElementById('status');
let resultBlob = null;
let currentFileName = '';
let timerInterval = null;
let startTime = 0;

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

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

async function handleFile(file) {
  if (!file.type.startsWith('image/')) { setStatus('Please upload an image file.', true); return; }

  currentFileName = file.name;

  // Generate job ID immediately and update URL
  const pendingId = [...crypto.getRandomValues(new Uint8Array(4))].map(b => b.toString(16).padStart(2,'0')).join('');
  history.replaceState(null, '', '?job=' + pendingId);

  // Show original preview
  const origUrl = URL.createObjectURL(file);
  origPreview.src = origUrl;
  resultOrig.src = origUrl;
  fileInfo.textContent = file.name + ' (' + formatSize(file.size) + ')';

  dropZone.style.display = 'none';
  processingArea.style.display = 'block';
  resultArea.style.display = 'none';
  setStatus('');

  // Start progress animation: ease to 97% over 25s, pause there until done
  progressFill.style.transition = 'none';
  progressFill.style.width = '0%';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      progressFill.style.transition = 'width 25s cubic-bezier(0.25, 0.1, 0.25, 1)';
      progressFill.style.width = '97%';
    });
  });

  startTime = performance.now();
  timerInterval = setInterval(() => {
    const s = ((performance.now() - startTime) / 1000).toFixed(1);
    elapsedEl.textContent = s + 's';
  }, 100);

  try {
    const form = new FormData();
    form.append('file', file);
    form.append('job_id', pendingId);
    const resp = await fetch('/remove-bg', { method: 'POST', body: form });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(text || 'Server error ' + resp.status);
    }

    resultBlob = await resp.blob();
    const procMs = resp.headers.get('X-Processing-Time-Ms');
    const jobId = resp.headers.get('X-Job-Id');
    const procSec = procMs ? (parseInt(procMs) / 1000).toFixed(1) : null;

    // Complete progress bar
    clearInterval(timerInterval);
    progressFill.style.transition = 'width 0.3s ease';
    progressFill.style.width = '100%';
    const wallSec = ((performance.now() - startTime) / 1000).toFixed(1);

    // Show side-by-side result
    resultImg.src = URL.createObjectURL(resultBlob);
    timeBadge.textContent = procSec ? procSec + 's' : wallSec + 's';

    processingArea.style.display = 'none';
    resultArea.style.display = 'block';

    downloadBtn.onclick = () => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(resultBlob);
      a.download = file.name.replace(/\\.[^.]+$/, '') + '-nobg.png';
      a.click();
    };

  } catch (err) {
    clearInterval(timerInterval);
    processingArea.style.display = 'none';
    setStatus('Error: ' + err.message, true);
    dropZone.style.display = '';
  }
}

function setStatus(msg, isErr) {
  status.textContent = msg;
  status.className = isErr ? 'error' : '';
}

function resetUI() {
  if (timerInterval) clearInterval(timerInterval);
  resultArea.style.display = 'none';
  processingArea.style.display = 'none';
  resultImg.src = '';
  resultOrig.src = '';
  origPreview.src = '';
  resultBlob = null;
  fileInput.value = '';
  progressFill.style.width = '0%';
  dropZone.style.display = '';
  setStatus('');
  history.replaceState(null, '', '/');
}

// Lightbox
const lightbox = document.getElementById('lightbox');
const lbImg = document.getElementById('lb-img');
const lbWrap = document.getElementById('lb-img-wrap');
const bgBar = document.getElementById('bg-bar');
let lbMode = null; // 'orig' or 'result'

const checkerDark = 'repeating-conic-gradient(#2a2a2a 0% 25%, #222 0% 50%) 50%/20px 20px';
const checkerLight = 'repeating-conic-gradient(#ddd 0% 25%, #fff 0% 50%) 50%/20px 20px';

function openLightbox(mode) {
  lbMode = mode;
  if (mode === 'orig') {
    lbImg.src = resultOrig.src;
    lbWrap.style.background = '#1a1a1a';
    bgBar.style.display = 'none';
  } else {
    lbImg.src = resultImg.src;
    lbWrap.style.background = checkerDark;
    bgBar.style.display = 'flex';
    // reset swatch selection
    bgBar.querySelectorAll('.bg-swatch').forEach(s => s.classList.remove('active'));
    bgBar.querySelector('[data-bg="checker-dark"]').classList.add('active');
  }
  lightbox.classList.add('active');
}

function closeLightbox() {
  lightbox.classList.remove('active');
  lbMode = null;
}

resultOrig.addEventListener('click', () => openLightbox('orig'));
resultImg.addEventListener('click', () => openLightbox('result'));
document.getElementById('lb-close').addEventListener('click', closeLightbox);
lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

bgBar.addEventListener('click', e => {
  const sw = e.target.closest('.bg-swatch');
  if (!sw) return;
  bgBar.querySelectorAll('.bg-swatch').forEach(s => s.classList.remove('active'));
  sw.classList.add('active');
  const bg = sw.dataset.bg;
  if (bg === 'checker-dark') lbWrap.style.background = checkerDark;
  else if (bg === 'checker-light') lbWrap.style.background = checkerLight;
  else lbWrap.style.background = bg;
});

// Load job from URL param on page load
(async function() {
  const params = new URLSearchParams(location.search);
  const jobId = params.get('job');
  if (!jobId) return;
  try {
    const meta = await fetch('/job/' + jobId).then(r => {
      if (!r.ok) throw new Error('not found');
      return r.json();
    });
    const ext = (meta.filename || 'image.png').split('.').pop();
    dropZone.style.display = 'none';

    if (meta.processingTimeMs != null) {
      // Completed job — show side-by-side result
      currentFileName = meta.filename;
      resultOrig.src = '/history/images/' + jobId + '_orig.' + ext;
      resultImg.src = '/history/images/' + jobId + '_result.png';
      const procSec = (meta.processingTimeMs / 1000).toFixed(1);
      timeBadge.textContent = procSec + 's';
      resultBlob = null;
      downloadBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = '/history/images/' + jobId + '_result.png';
        a.download = (meta.filename || 'image').replace(/\\.[^.]+$/, '') + '-nobg.png';
        a.click();
      };
      resultArea.style.display = 'block';
    } else {
      // Pending job — show original + spinner, poll until done
      origPreview.src = '/history/images/' + jobId + '_orig.' + ext;
      resultOrig.src = origPreview.src;
      fileInfo.textContent = meta.filename || 'image';
      processingArea.style.display = 'block';
      progressFill.style.transition = 'none';
      progressFill.style.width = '0%';
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          progressFill.style.transition = 'width 25s cubic-bezier(0.25, 0.1, 0.25, 1)';
          progressFill.style.width = '97%';
        });
      });
      const pollStart = performance.now();
      const pollTimer = setInterval(() => {
        elapsedEl.textContent = ((performance.now() - pollStart) / 1000).toFixed(1) + 's';
      }, 100);
      const poll = async () => {
        try {
          const m = await fetch('/job/' + jobId).then(r => r.json());
          if (m.processingTimeMs != null) {
            clearInterval(pollTimer);
            progressFill.style.transition = 'width 0.3s ease';
            progressFill.style.width = '100%';
            resultImg.src = '/history/images/' + jobId + '_result.png';
            const procSec = (m.processingTimeMs / 1000).toFixed(1);
            timeBadge.textContent = procSec + 's';
            resultBlob = null;
            downloadBtn.onclick = () => {
              const a = document.createElement('a');
              a.href = '/history/images/' + jobId + '_result.png';
              a.download = (m.filename || 'image').replace(/\\.[^.]+$/, '') + '-nobg.png';
              a.click();
            };
            processingArea.style.display = 'none';
            resultArea.style.display = 'block';
          } else {
            setTimeout(poll, 1000);
          }
        } catch {
          setTimeout(poll, 2000);
        }
      };
      setTimeout(poll, 1000);
    }
  } catch {
    // job not found — show normal UI
    history.replaceState(null, '', '/');
  }
})();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML


# ─── History UI ─────────────────────────────────────────────────────────────

HISTORY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BG Remove — History</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f0f0f; color: #e0e0e0; min-height: 100vh; padding: 2rem; }
  .header { display: flex; justify-content: space-between; align-items: center;
             max-width: 1200px; margin: 0 auto 2rem; }
  h1 { font-size: 1.5rem; }
  .header-right { display: flex; gap: 0.75rem; align-items: center; }
  .count { color: #888; font-size: 0.9rem; }
  .btn {
    background: #646cff; color: #fff; border: none; padding: 0.5rem 1rem;
    border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600;
    transition: background 0.15s;
  }
  .btn:hover { background: #535bf2; }
  .btn.danger { background: #c0392b; }
  .btn.danger:hover { background: #e74c3c; }
  .btn.secondary { background: #333; }
  .btn.secondary:hover { background: #444; }
  #grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem; max-width: 1200px; margin: 0 auto;
  }
  .card {
    background: #1a1a1a; border-radius: 12px; overflow: hidden; cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
  }
  .card:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(100,108,255,0.15); }
  .card .thumb-row {
    display: grid; grid-template-columns: 1fr 1fr; height: 160px;
  }
  .card .thumb-row .half {
    display: flex; justify-content: center; align-items: center; overflow: hidden;
  }
  .card .thumb-row .half.checker {
    background: repeating-conic-gradient(#2a2a2a 0% 25%, #222 0% 50%) 50%/16px 16px;
  }
  .card .thumb-row img { width: 100%; height: 100%; object-fit: cover; }
  .card .card-info {
    padding: 0.75rem 1rem; border-top: 1px solid #2a2a2a;
    display: flex; justify-content: space-between; align-items: center;
  }
  .card .card-info .fname { font-size: 0.85rem; color: #ccc;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 60%; }
  .card .card-info .meta { font-size: 0.75rem; color: #666; }
  .card .card-info .meta .time { color: #646cff; font-weight: 600; }
  #empty { text-align: center; color: #555; padding: 4rem; font-size: 1.1rem; }

  /* Lightbox */
  #lightbox {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85);
    z-index: 100; justify-content: center; align-items: center;
  }
  #lightbox.active { display: flex; }
  #lightbox .lb-content {
    max-width: 90vw; max-height: 90vh; display: grid;
    grid-template-columns: 1fr 1fr; gap: 1rem; background: #1a1a1a;
    border-radius: 16px; overflow: hidden;
  }
  @media (max-width: 700px) { #lightbox .lb-content { grid-template-columns: 1fr; } }
  #lightbox .lb-half { display: flex; flex-direction: column; }
  #lightbox .lb-half .lb-label {
    padding: 0.75rem 1rem; font-weight: 600; font-size: 0.9rem;
    border-bottom: 1px solid #2a2a2a; text-align: center;
  }
  #lightbox .lb-half .lb-img-wrap {
    flex: 1; display: flex; justify-content: center; align-items: center;
    padding: 1rem; min-height: 300px;
  }
  #lightbox .lb-half .lb-img-wrap.checker {
    background: repeating-conic-gradient(#2a2a2a 0% 25%, #222 0% 50%) 50%/20px 20px;
  }
  #lightbox .lb-half img { max-width: 100%; max-height: 70vh; border-radius: 6px; }
  #lightbox .lb-close {
    position: absolute; top: 1rem; right: 1.5rem; font-size: 2rem;
    color: #888; cursor: pointer; background: none; border: none;
  }
  #lightbox .lb-close:hover { color: #fff; }
  #lightbox .lb-footer {
    grid-column: 1 / -1; padding: 0.75rem 1rem; border-top: 1px solid #2a2a2a;
    display: flex; justify-content: space-between; align-items: center;
  }
  #lightbox .lb-footer .lb-fname { color: #ccc; font-size: 0.9rem; }
  #lightbox .lb-footer .lb-meta { color: #888; font-size: 0.8rem; }
</style>
</head>
<body>

<div class="header">
  <h1>History</h1>
  <div class="header-right">
    <span class="count" id="count"></span>
    <button class="btn secondary" onclick="location.href='/'">Back</button>
    <button class="btn danger" id="clear-btn" style="display:none" onclick="clearHistory()">Clear All</button>
  </div>
</div>

<div id="empty">No background removals yet.</div>
<div id="grid"></div>

<div id="lightbox">
  <button class="lb-close" onclick="closeLightbox()">&times;</button>
  <div class="lb-content">
    <div class="lb-half">
      <div class="lb-label">Original</div>
      <div class="lb-img-wrap"><img id="lb-orig"></div>
    </div>
    <div class="lb-half">
      <div class="lb-label">Result</div>
      <div class="lb-img-wrap checker"><img id="lb-result"></div>
    </div>
    <div class="lb-footer">
      <span class="lb-fname" id="lb-fname"></span>
      <span class="lb-meta" id="lb-meta"></span>
    </div>
  </div>
</div>

<script>
let items = [];

async function loadHistory() {
  try {
    const resp = await fetch('/history/data');
    items = await resp.json();
    items.reverse(); // newest first
  } catch { items = []; }
  render();
}

function render() {
  const grid = document.getElementById('grid');
  const empty = document.getElementById('empty');
  const count = document.getElementById('count');
  const clearBtn = document.getElementById('clear-btn');

  count.textContent = items.length + ' jobs';
  clearBtn.style.display = items.length ? 'inline-block' : 'none';

  if (!items.length) {
    grid.innerHTML = '';
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';

  grid.innerHTML = items.map((it, i) => {
    const ext = it.filename.split('.').pop() || 'png';
    return `
      <div class="card" onclick="openLightbox(${i})">
        <div class="thumb-row">
          <div class="half"><img src="/history/images/${it.id}_orig.${ext}" loading="lazy"></div>
          <div class="half checker"><img src="/history/images/${it.id}_result.png" loading="lazy"></div>
        </div>
        <div class="card-info">
          <span class="fname">${esc(it.filename)}</span>
          <span class="meta"><span class="time">${(it.processingTimeMs/1000).toFixed(1)}s</span> &middot; ${fmtDate(it.timestamp)} &middot; <a href="/?job=${it.id}" style="color:#646cff;text-decoration:none" onclick="event.stopPropagation()">view</a></span>
        </div>
      </div>`;
  }).join('');
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function fmtDate(ts) {
  try {
    const d = new Date(ts);
    const now = new Date();
    if (d.toDateString() === now.toDateString()) return d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
    return d.toLocaleDateString([], {month:'short',day:'numeric'}) + ' ' + d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  } catch { return ''; }
}

function openLightbox(i) {
  const it = items[i];
  const ext = it.filename.split('.').pop() || 'png';
  document.getElementById('lb-orig').src = '/history/images/' + it.id + '_orig.' + ext;
  document.getElementById('lb-result').src = '/history/images/' + it.id + '_result.png';
  document.getElementById('lb-fname').textContent = it.filename;
  document.getElementById('lb-meta').textContent = (it.processingTimeMs/1000).toFixed(1) + 's · ' + fmtDate(it.timestamp);
  document.getElementById('lightbox').classList.add('active');
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('active');
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });
document.getElementById('lightbox').addEventListener('click', e => {
  if (e.target.id === 'lightbox') closeLightbox();
});

async function clearHistory() {
  if (!confirm('Delete all history? This cannot be undone.')) return;
  await fetch('/history/clear', { method: 'POST' });
  items = [];
  render();
}

loadHistory();
</script>
</body>
</html>"""


@app.get("/history", response_class=HTMLResponse)
def history_ui():
    return HISTORY_HTML


@app.get("/history/data")
def history_data():
    return load_history()


@app.get("/history/images/{filename}")
def history_image(filename: str):
    filepath = HISTORY_DIR / filename
    if not filepath.exists() or ".." in filename:
        raise HTTPException(status_code=404)
    ext = filepath.suffix.lower()
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    return Response(content=filepath.read_bytes(), media_type=media.get(ext, "application/octet-stream"))


@app.post("/history/clear")
def history_clear():
    for f in HISTORY_DIR.iterdir():
        try:
            f.unlink()
        except OSError:
            pass
    return {"ok": True}


# ─── API ────────────────────────────────────────────────────────────────────

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
        "device": str(_device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    job_id: Optional[str] = Form(None),
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

    # Save original for history — use client-provided ID if valid
    if job_id and len(job_id) == 8 and all(c in '0123456789abcdef' for c in job_id.lower()):
        job_id = job_id.lower()
    else:
        job_id = uuid.uuid4().hex[:8]
    orig_ext = (file.filename or "image.png").rsplit(".", 1)[-1].lower()
    if orig_ext not in ("png", "jpg", "jpeg", "webp"):
        orig_ext = "png"
    orig_path = HISTORY_DIR / f"{job_id}_orig.{orig_ext}"
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    orig_path.write_bytes(data)

    # Record pending history entry (so ?job= pages show the original + spinner)
    add_history_entry(job_id, file.filename or "image.png")

    # Inference
    t0 = time.time()
    input_tensor = _transform(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        preds = _model(input_tensor)[-1].sigmoid().cpu()
    mask = transforms.ToPILImage()(preds[0].squeeze()).resize(image.size)
    image.putalpha(mask)
    processing_ms = int((time.time() - t0) * 1000)

    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    result_bytes = buf.getvalue()

    # Save result for history
    result_path = HISTORY_DIR / f"{job_id}_result.png"
    result_path.write_bytes(result_bytes)

    # Update history entry with processing time
    complete_history_entry(job_id, processing_ms)

    return Response(
        content=result_bytes,
        media_type="image/png",
        headers={"X-Processing-Time-Ms": str(processing_ms), "X-Job-Id": job_id},
    )


@app.get("/job/{job_id}")
def job_detail(job_id: str):
    """Return metadata for a specific job."""
    for entry in load_history():
        if entry.get("id") == job_id:
            return entry
    raise HTTPException(status_code=404, detail="Job not found")
