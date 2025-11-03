"""
app.py - X-Ray Defacer FastAPI application

Features:
- Upload an X-ray image
- Call Roboflow hosted inference API with MODEL_URL & ROBOFLOW_API_KEY
- Filter predictions by confidence threshold
- Create a mask from detections and either blur or blackout those regions
- Return before/after images (base64) + metadata to template
- Structured logs with image size, dimensions, predictions, latency
- Optional saving of outputs to ./output/
- A small route to respond to Chrome DevTools probe (silence 404 noise)
"""

import os
import time
import logging
from io import BytesIO
from typing import List, Dict, Any, Tuple
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFilter
from fastapi.middleware.cors import CORSMiddleware
import base64

# ---------------------------
# Configuration & Logging
# ---------------------------
load_dotenv()

MODEL_URL = os.getenv("MODEL_URL")  # e.g. "https://detect.roboflow.com/xray-defacer/10"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.34"))  # default 0.34
SAVE_OUTPUTS = os.getenv("SAVE_OUTPUTS", "true").lower() in ("1", "true", "yes")

if not MODEL_URL or not ROBOFLOW_API_KEY:
    raise RuntimeError("Missing MODEL_URL or ROBOFLOW_API_KEY in environment or .env")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("xray-defacer")



# ---------------------------
# App & Templates
# ---------------------------
app = FastAPI(title="X-Ray Defacer", version="1.0.0")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Helpers
# ---------------------------
def call_roboflow(image_bytes: bytes, timeout: int = 60) -> Dict[str, Any]:
    """Call Roboflow hosted inference endpoint and return parsed JSON."""
    url = f"{MODEL_URL}?api_key={ROBOFLOW_API_KEY}"
    resp = requests.post(
        url,
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def parse_predictions(rf_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse Roboflow JSON to a standard list of predictions with keys:
      - class, confidence (0..1), x, y, width, height (pixel coords OR center-based)
    This function attempts to be defensive about different response shapes.
    """
    preds = []
    candidates = rf_json.get("predictions") or rf_json.get("preds") or []
    # support case where top-level 'predictions' is nested
    if isinstance(candidates, dict) and "predictions" in candidates:
        candidates = candidates["predictions"]

    for p in candidates:
        try:
            cls = p.get("class") or p.get("label") or p.get("type") or "unknown"
            conf = p.get("confidence") or p.get("score") or p.get("probability") or 0.0
            # Some APIs return confidence as percentage, handle both
            if conf > 1 and conf <= 100:
                conf = conf / 100.0
            x = p.get("x") or p.get("center_x") or p.get("cx")
            y = p.get("y") or p.get("center_y") or p.get("cy")
            width = p.get("width") or p.get("w")
            height = p.get("height") or p.get("h")

            # if coordinates are normalized (0..1) we cannot tell here; assume pixel for Roboflow
            preds.append({
                "class": cls,
                "confidence": float(conf),
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "raw": p,
            })
        except Exception as e:
            logger.warning("Failed to parse a prediction entry: %s", e)
    return preds


def filter_by_confidence(preds: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    return [p for p in preds if (p.get("confidence") or 0.0) >= threshold]


def build_mask_from_preds(img_w: int, img_h: int, preds: List[Dict[str, Any]]) -> Image.Image:
    """
    Create grayscale mask (mode "L") where white (255) marks area to deface.
    Assumes predictions provide center-based boxes (x,y,width,height).
    """
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    for p in preds:
        x = p.get("x")
        y = p.get("y")
        w = p.get("width")
        h = p.get("height")
        if x is None or y is None or w is None or h is None:
            # skip incomplete boxes
            continue
        # convert center-based to corners
        left = max(0, x - w / 2)
        top = max(0, y - h / 2)
        right = min(img_w, x + w / 2)
        bottom = min(img_h, y + h / 2)
        draw.rectangle([left, top, right, bottom], fill=255)
    return mask


def blur_with_mask(image: Image.Image, mask: Image.Image, radius: int = 18) -> Image.Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return Image.composite(blurred, image, mask)


def blackout_with_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    black = Image.new("RGB", image.size, (0, 0, 0))
    return Image.composite(black, image, mask)


def to_base64_jpeg(img: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_image(img: Image.Image, filename: str) -> str:
    out_path = OUTPUT_DIR / filename
    img.save(out_path, format="JPEG", quality=90)
    return str(out_path.resolve())


# ---------------------------
# Routes
# ---------------------------
@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def ignore_chrome_devtools_probe():
    """
    A tiny route Chrome DevTools may probe. Returning a valid JSON lets us avoid noisy 404 logs.
    """
    return {}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/deface", response_class=HTMLResponse)
async def deface(request: Request, file: UploadFile = File(...)):
    """
    Accept an uploaded image, call Roboflow inference, create mask from
    detections (filtered by CONFIDENCE_THRESHOLD), then blackout regions and return results.
    """
    start_time = time.time()

    # read file
    raw = await file.read()
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image uploaded: {e}")

    width, height = image.size
    logger.info("Received file=%s size=%.1fKB dims=%dx%d", file.filename, len(raw) / 1024.0, width, height)

    # call Roboflow
    try:
        rf_json = call_roboflow(raw)
    except requests.HTTPError as e:
        logger.error("Roboflow request failed: %s", e)
        raise HTTPException(status_code=502, detail="Inference call to Roboflow failed")
    except Exception as e:
        logger.exception("Unexpected error calling Roboflow: %s", e)
        raise HTTPException(status_code=502, detail="Inference service error")

    # parse & filter
    preds = parse_predictions(rf_json)
    logger.info("Raw predictions count=%d", len(preds))
    filtered = filter_by_confidence(preds, CONFIDENCE_THRESHOLD)
    logger.info("Filtered predictions (confidence >= %.2f): %d", CONFIDENCE_THRESHOLD, len(filtered))

    for i, p in enumerate(filtered, start=1):
        logger.info("  %02d: class=%s conf=%.3f raw=%s", i, p.get("class"), p.get("confidence"), p.get("raw"))

    # build mask and deface
    mask = build_mask_from_preds(width, height, filtered)
    # Choose blackout to hard-anonymize (change to blur_with_mask to blur)
    defaced = blackout_with_mask(image, mask)

    # optionally save to disk
    if SAVE_OUTPUTS:
        stamp = int(time.time())
        in_name = Path(file.filename).stem if file.filename else "upload"
        out_name = f"{in_name}_defaced_{stamp}.jpg"
        saved_path = save_image(defaced, out_name)
        logger.info("Saved defaced image: %s", saved_path)

    # encode results
    before_b64 = to_base64_jpeg(image)
    after_b64 = to_base64_jpeg(defaced)

    latency = time.time() - start_time
    avg_conf = round(sum((p.get("confidence") or 0.0) for p in filtered) / max(len(filtered), 1), 4) if filtered else 0.0

    # pass metadata to template (you can render these in index.html)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "before": before_b64,
            "after": after_b64,
            "num_preds": len(filtered),
            "avg_conf": avg_conf,
            "model_url": MODEL_URL,
            "latency": round(latency, 3),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        },
    )


# ---------------------------
# Small API endpoint for quick testing (JSON)
# ---------------------------
@app.post("/api/deface", response_class=JSONResponse)
async def api_deface(file: UploadFile = File(...)):
    """
    JSON endpoint for automated testing or integration.
    Returns detection metadata and a base64-encoded defaced image.
    """
    start_time = time.time()

    # read and validate file
    raw = await file.read()
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image uploaded: {e}")

    width, height = image.size
    rf_json = call_roboflow(raw)
    preds = parse_predictions(rf_json)
    filtered = filter_by_confidence(preds, CONFIDENCE_THRESHOLD)

    mask = build_mask_from_preds(width, height, filtered)
    defaced = blackout_with_mask(image, mask)  # or blur_with_mask

    encoded_img = to_base64_jpeg(defaced)
    latency = round(time.time() - start_time, 3)
    avg_conf = round(sum((p.get("confidence") or 0.0) for p in filtered) / max(len(filtered), 1), 4) if filtered else 0.0

    cleaned_preds = [
        {
            "class": p.get("class", "unknown"),
            "confidence": round(p.get("confidence", 0.0), 4),
            "x": p.get("x"),
            "y": p.get("y"),
            "width": p.get("width"),
            "height": p.get("height"),
        }
        for p in filtered
    ]

    return JSONResponse(
        {
            "num_preds": len(filtered),
            "avg_conf": avg_conf,
            "predictions": cleaned_preds,
            "latency": latency,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "image": encoded_img,
        }
    )


# EOF
