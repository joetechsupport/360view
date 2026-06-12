"""
OMMA Stitch Service
FastAPI + OpenCV panorama stitching backend.

Endpoints:
  GET  /health
  GET  /api/health
  POST /api/stitch
"""

import os
import io
import base64
import time
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ExifTags
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("omma-stitch")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
PORT = int(os.environ.get("PORT", 8000))
API_KEY = os.environ.get("OMMA_API_KEY", "")          # empty = no auth
MAX_IMAGE_DIMENSION = int(os.environ.get("MAX_IMAGE_DIMENSION", 3000))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", 40))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.70))
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "*"
).split(",")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OMMA Stitch Service",
    version="1.0.0",
    description="OpenCV panorama stitching backend for OMMA 360 Builder.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def verify_api_key(request: Request):
    """If OMMA_API_KEY is set, require it in Authorization header."""
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ImagePayload(BaseModel):
    index: int
    filename: str = "image.jpg"
    data: str                        # base64-encoded image bytes
    mimeType: str = "image/jpeg"

class StitchOptions(BaseModel):
    outputFormat: str = "jpeg"
    outputQuality: float = 0.92
    targetProjection: str = "equirectangular"

class StitchRequest(BaseModel):
    roomId: str
    roomName: str = "Room"
    imageCount: int = 0
    images: list[ImagePayload]
    options: StitchOptions = Field(default_factory=StitchOptions)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fix_orientation(pil_img: Image.Image) -> Image.Image:
    """Rotate image according to EXIF orientation tag."""
    try:
        exif = pil_img._getexif()
        if exif is None:
            return pil_img
        orientation_key = next(
            k for k, v in ExifTags.TAGS.items() if v == "Orientation"
        )
        orientation = exif.get(orientation_key)
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            pil_img = pil_img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass
    return pil_img


def decode_image(payload: ImagePayload) -> np.ndarray:
    """
    Decode a base64 image payload into an OpenCV BGR ndarray.
    Raises ValueError on any failure.
    """
    try:
        # Strip data-URI prefix if present
        data = payload.data
        if "," in data:
            data = data.split(",", 1)[1]

        raw = base64.b64decode(data)
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        pil_img = fix_orientation(pil_img)

        # Resize if oversized
        w, h = pil_img.size
        if max(w, h) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(w, h)
            pil_img = pil_img.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS,
            )
            log.info(
                "Resized %s: %dx%d → %dx%d",
                payload.filename, w, h,
                pil_img.width, pil_img.height,
            )

        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    except Exception as exc:
        raise ValueError(f"Cannot decode {payload.filename}: {exc}") from exc


def encode_jpeg(bgr: np.ndarray, quality: int = 92) -> str:
    """Encode an OpenCV BGR image to a base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def make_thumbnail(bgr: np.ndarray, max_width: int = 512) -> str:
    """Down-scale and encode a thumbnail."""
    h, w = bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        bgr = cv2.resize(bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return encode_jpeg(bgr, quality=80)


def score_panorama(pano: np.ndarray) -> dict:
    """
    Compute a confidence score and quality signals for the stitched panorama.

    Returns a dict with:
      confidence  (float 0-1)
      warnings    (list[str])
      gates       (dict)
    """
    warnings = []
    h, w = pano.shape[:2]

    # Gate 1: minimum dimensions
    min_ok = w >= 1024 and h >= 256
    if not min_ok:
        warnings.append(f"Output too small: {w}x{h}")

    # Gate 2: aspect ratio close to 2:1
    aspect = w / h if h else 0
    aspect_ok = 1.5 <= aspect <= 3.0
    if not aspect_ok:
        warnings.append(
            f"Aspect ratio {aspect:.2f} is not close to 2:1 — may not be equirectangular."
        )

    # Gate 3: black border fraction (indicates stitching artifacts)
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray < 10)
    total_pixels = h * w
    black_fraction = black_pixels / total_pixels if total_pixels else 1.0
    border_ok = black_fraction < 0.20
    if not border_ok:
        warnings.append(
            f"Panorama has {black_fraction:.1%} black pixels — likely incomplete overlap."
        )

    # Gate 4: edge density (low = blurry / failed stitch)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / total_pixels if total_pixels else 0.0
    edge_ok = edge_density > 0.01
    if not edge_ok:
        warnings.append("Very low edge density — panorama may be blank or blurry.")

    # Composite confidence
    score = sum([min_ok, aspect_ok, border_ok, edge_ok]) / 4.0
    # Penalise heavily for bad borders
    if black_fraction > 0.40:
        score *= 0.5

    gates = {
        "minDimensions": min_ok,
        "aspectRatio": aspect_ok,
        "borderArtifacts": border_ok,
        "edgeDensity": edge_ok,
    }

    return {
        "confidence": round(score, 4),
        "warnings": warnings,
        "gates": gates,
        "width": w,
        "height": h,
        "aspectRatio": round(aspect, 3),
        "blackFraction": round(black_fraction, 4),
    }


def run_opencv_stitch(images: list[np.ndarray]) -> tuple[bool, Optional[np.ndarray], str]:
    """
    Attempt panorama stitching with OpenCV.

    Returns:
      (success: bool, panorama: ndarray | None, message: str)

    Tries PANORAMA mode first, then falls back to SCANS mode.
    """
    modes = [
        ("PANORAMA", cv2.Stitcher_PANORAMA),
        ("SCANS", cv2.Stitcher_SCANS),
    ]

    for label, mode in modes:
        try:
            log.info("Attempting OpenCV stitch in %s mode with %d images…", label, len(images))
            stitcher = cv2.Stitcher.create(mode)
            status, pano = stitcher.stitch(images)

            if status == cv2.Stitcher_OK and pano is not None and pano.size > 0:
                log.info("Stitch succeeded in %s mode.", label)
                return True, pano, f"Stitched successfully in {label} mode."

            code_map = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough overlapping features — add more images.",
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed — poor overlap or too few keypoints.",
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed — try different images.",
            }
            reason = code_map.get(status, f"OpenCV Stitcher returned error code {status}.")
            log.warning("%s mode failed: %s", label, reason)

        except cv2.error as exc:
            log.warning("%s mode raised cv2.error: %s", label, exc)
            reason = str(exc)

    return False, None, reason   # noqa: F821 (last reason from loop)

# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------
HEALTH_BODY = {
    "ok": True,
    "service": "omma-stitch-service",
    "engine": "opencv",
    "version": "1.0.0",
    "opencvVersion": cv2.__version__,
    "maxImages": MAX_IMAGES,
    "maxImageDimension": MAX_IMAGE_DIMENSION,
    "confidenceThreshold": CONFIDENCE_THRESHOLD,
    "auth": bool(API_KEY),
}


@app.get("/health")
async def health():
    return HEALTH_BODY


@app.get("/api/health")
async def api_health():
    return HEALTH_BODY

# ---------------------------------------------------------------------------
# Stitch endpoint
# ---------------------------------------------------------------------------
@app.post("/api/stitch")
async def stitch(
    req: StitchRequest,
    _: None = Depends(verify_api_key),
):
    t_start = time.monotonic()
    room_id = req.roomId
    log.info("Stitch request: roomId=%s images=%d", room_id, len(req.images))

    # ── Validate image count ──────────────────────────────────────────────
    if len(req.images) < 2:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "isTrue360": False,
                "engine": "opencv",
                "roomId": room_id,
                "error": "At least 2 images are required for stitching.",
                "warnings": [],
            },
        )

    if len(req.images) > MAX_IMAGES:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "isTrue360": False,
                "engine": "opencv",
                "roomId": room_id,
                "error": f"Too many images: {len(req.images)} (max {MAX_IMAGES}).",
                "warnings": [],
            },
        )

    # ── Decode images ─────────────────────────────────────────────────────
    cv_images: list[np.ndarray] = []
    decode_warnings: list[str] = []

    for payload in sorted(req.images, key=lambda x: x.index):
        try:
            img = decode_image(payload)
            cv_images.append(img)
        except ValueError as exc:
            decode_warnings.append(str(exc))
            log.warning("Skipping %s: %s", payload.filename, exc)

    if len(cv_images) < 2:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "isTrue360": False,
                "engine": "opencv",
                "roomId": room_id,
                "error": f"Only {len(cv_images)} image(s) decoded successfully — need at least 2.",
                "warnings": decode_warnings,
            },
        )

    # ── Run stitcher ──────────────────────────────────────────────────────
    ok, pano, stitch_msg = run_opencv_stitch(cv_images)

    if not ok or pano is None:
        elapsed = round((time.monotonic() - t_start) * 1000)
        log.error("Stitch failed for %s: %s", room_id, stitch_msg)
        return {
            "success": False,
            "isTrue360": False,
            "engine": "opencv",
            "roomId": room_id,
            "error": stitch_msg,
            "warnings": decode_warnings,
            "processingTimeMs": elapsed,
        }

    # ── Quality gates ─────────────────────────────────────────────────────
    quality = score_panorama(pano)
    all_warnings = decode_warnings + quality["warnings"]
    confidence = quality["confidence"]

    is_true_360 = (
        confidence >= CONFIDENCE_THRESHOLD
        and quality["gates"]["minDimensions"]
        and quality["gates"]["aspectRatio"]
        and quality["gates"]["borderArtifacts"]
    )

    if not is_true_360:
        log.warning(
            "Quality gates failed for %s: confidence=%.2f gates=%s",
            room_id, confidence, quality["gates"],
        )

    # ── Encode outputs ────────────────────────────────────────────────────
    jpeg_quality = max(1, min(100, int(req.options.outputQuality * 100)))
    panorama_b64 = encode_jpeg(pano, quality=jpeg_quality)
    thumbnail_b64 = make_thumbnail(pano)

    elapsed = round((time.monotonic() - t_start) * 1000)
    log.info(
        "Done: roomId=%s isTrue360=%s confidence=%.2f time=%dms",
        room_id, is_true_360, confidence, elapsed,
    )

    return {
        "success": True,
        "isTrue360": is_true_360,
        "engine": "opencv",
        "roomId": room_id,
        "panoramaBase64": panorama_b64,
        "thumbnailBase64": thumbnail_b64,
        "width": quality["width"],
        "height": quality["height"],
        "aspectRatio": quality["aspectRatio"],
        "confidence": confidence,
        "qualityGates": quality["gates"],
        "warnings": all_warnings,
        "processingTimeMs": elapsed,
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
