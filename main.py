"""
OMMA 360 Builder — OpenCV Stitch Service
=========================================
Production stitching backend for OMMA.
Receives overlapping room photos, runs OpenCV Stitcher,
returns a real equirectangular panorama or honest failure.

AI RECONSTRUCT MODE:
When stitch confidence is low or coverage is incomplete,
the ai_reconstruct pipeline preserves successful regions,
estimates room geometry, and synthesizes missing areas
using geometric inpainting. AI-generated pixels are tagged
internally and never modify original source photographs.

MEMORY-OPTIMIZED for Render free/starter (512 MB).
All images are immediately downsampled on decode.
No full-resolution copies are ever held in memory.
"""

import os
import io
import gc
import math
import base64
import time
import hashlib
import logging
from typing import Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image, ExifTags
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════
#  EMBEDDED MANAGED FILES (for deploy endpoint)
#  These files aren't COPY'd into Docker, so we
#  embed them here so the deploy endpoint can push
#  the complete set to GitHub.
# ═══════════════════════════════════════════════════

EMBEDDED_DOCKERFILE = """FROM python:3.11-slim

# OpenCV headless system dependencies (minimal set)
# Note: libgl1-mesa-glx removed — no longer available on current Debian base.
# libgl1 provides the same OpenGL runtime needed by OpenCV headless.
RUN apt-get update && apt-get install -y --no-install-recommends \\
    libglib2.0-0 \\
    libgl1 \\
    libgomp1 \\
 && rm -rf /var/lib/apt/lists/* \\
 && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Render provides PORT via environment
ENV PORT=8000

EXPOSE ${PORT}

CMD ["python", "main.py"]
""".lstrip()

EMBEDDED_RENDER_YAML = """services:
  - type: web
    name: omma-stitch-service
    runtime: docker
    plan: starter
    healthCheckPath: /health
    envVars:
      - key: PORT
        value: 8000
      - key: MAX_IMAGES
        value: 30
      - key: MAX_PAYLOAD_MB
        value: 100
      - key: JPEG_QUALITY
        value: 85
      - key: OMMA_API_KEY
        value: ""
""".lstrip()

EMBEDDED_README = """# OMMA Stitch Service — OpenCV Backend + AI Reconstruct

Production stitching backend for the OMMA 360 Builder.
**Memory-optimized for Render starter tier (512 MB).**

## Endpoints

- `GET /health` or `GET /api/health` — Health check with capabilities
- `POST /api/stitch` — OpenCV panorama stitching
- `POST /api/ai-reconstruct` — AI-assisted reconstruction for incomplete coverage
- `POST /api/deploy-managed-backend` — Push managed files to GitHub

## Deploy to Render

1. Push this repo to GitHub
2. Create a Web Service on Render pointing to the repo
3. Render auto-detects the Dockerfile
4. Set `OMMA_API_KEY` in Render dashboard if desired

## AI Reconstruct Mode

When construction photography has incomplete overlap:
1. Attempts OpenCV stitch (preserves all original pixels)
2. Detects coverage gaps
3. Estimates room geometry from covered regions
4. Pads to equirectangular if needed
5. Fills missing areas using geometric inpainting
6. Returns coverage mask for viewer overlay

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | 8000 | Server port |
| `MAX_IMAGES` | 30 | Max images per request |
| `MAX_PAYLOAD_MB` | 100 | Max request body size |
| `JPEG_QUALITY` | 85 | Output JPEG quality |
| `OMMA_API_KEY` | "" | Bearer token (optional) |
| `GITHUB_TOKEN` | "" | For deploy endpoint (optional) |
""".lstrip()

# ─── Logging ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("omma-stitch")

# ─── Config ───
PORT = int(os.environ.get("PORT", 8000))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", 50))
MAX_PAYLOAD_MB = int(os.environ.get("MAX_PAYLOAD_MB", 200))
API_KEY = os.environ.get("OMMA_API_KEY", "")
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", 85))
THUMBNAIL_WIDTH = int(os.environ.get("THUMBNAIL_WIDTH", 480))

# ─── MEMORY BUDGET ───
# Render free tier: 512 MB total, ~400 MB usable after OS + Python + OpenCV
# Each 1600x1200 BGR image ≈ 5.5 MB in RAM
# 12 images × 5.5 MB = 66 MB for source set
# OpenCV Stitcher internally creates ~2-3x copies = ~130-200 MB
# Panorama output + encoding = ~30-50 MB
# Total peak ≈ 300 MB — safely under 400 MB usable
#
# At 1200px max: 12 images × 4 MB = 48 MB source, ~250 MB peak
# At 800px max for 20+ images: keeps peak under 350 MB

MAX_DIM_FEW = 1600    # max dimension when <= 8 images
MAX_DIM_MEDIUM = 1200  # max dimension when 9-15 images
MAX_DIM_MANY = 800     # max dimension when > 15 images

def get_max_dim(image_count: int) -> int:
    """Scale down max resolution based on image count to stay within memory budget."""
    if image_count <= 8:
        return MAX_DIM_FEW
    elif image_count <= 15:
        return MAX_DIM_MEDIUM
    else:
        return MAX_DIM_MANY


# ─── Models ───

class StitchImageInput(BaseModel):
    index: int
    filename: str = ""
    data: str  # base64 encoded
    mimeType: str = "image/jpeg"

class StitchOptions(BaseModel):
    outputFormat: str = "jpeg"
    outputQuality: float = 0.85
    targetProjection: str = "equirectangular"

class StitchRequest(BaseModel):
    roomId: str
    roomName: str = ""
    imageCount: int = 0
    images: list[StitchImageInput]
    options: StitchOptions = Field(default_factory=StitchOptions)

class StitchResponse(BaseModel):
    success: bool
    isTrue360: bool
    isPreviewReady: bool = False      # True when stitch succeeds but isn't equirectangular
    engine: str = "opencv"
    roomId: str = ""
    panoramaBase64: Optional[str] = None
    thumbnailBase64: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    confidence: Optional[float] = None
    aspectRatio: Optional[float] = None
    isEquirectangular: Optional[bool] = None
    warnings: list[str] = []
    error: Optional[str] = None
    processingTimeMs: Optional[int] = None
    # AI Reconstruct fields
    aiReconstructed: bool = False
    originalCoveragePercent: Optional[float] = None  # 0-100
    aiGeneratedPercent: Optional[float] = None        # 0-100
    coverageMaskBase64: Optional[str] = None          # grayscale mask: white=original, black=AI
    reconstructionMethod: Optional[str] = None


# ─── App ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"OMMA Stitch Service starting on port {PORT}")
    logger.info(f"Memory-optimized mode: {MAX_DIM_FEW}px/≤8img, {MAX_DIM_MEDIUM}px/≤15img, {MAX_DIM_MANY}px/>15img")
    logger.info(f"Max images per request: {MAX_IMAGES}")
    logger.info(f"API key required: {bool(API_KEY)}")
    logger.info(f"OpenCV version: {cv2.__version__}")
    yield
    logger.info("OMMA Stitch Service shutting down")

app = FastAPI(
    title="OMMA Stitch Service",
    version="1.2.0",
    lifespan=lifespan
)

# CORS — allow OMMA frontend from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Auth Middleware ───

@app.middleware("http")
async def check_auth(request: Request, call_next):
    if request.url.path in ("/health", "/api/health", "/") or request.method == "OPTIONS":
        return await call_next(request)

    if API_KEY:
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.replace("Bearer ", "").strip()
        if token != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"success": False, "error": "Invalid or missing API key"}
            )

    return await call_next(request)


# ─── Payload Size Limit ───

@app.middleware("http")
async def limit_payload(request: Request, call_next):
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_PAYLOAD_MB * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={"success": False, "error": f"Payload too large. Max {MAX_PAYLOAD_MB}MB."}
            )
    return await call_next(request)


# ─── Image Processing ───

def fix_exif_orientation(pil_image: Image.Image) -> Image.Image:
    """Rotate image based on EXIF orientation tag."""
    try:
        exif = pil_image._getexif()
        if exif is None:
            return pil_image

        orientation_key = None
        for key, val in ExifTags.TAGS.items():
            if val == "Orientation":
                orientation_key = key
                break

        if orientation_key is None or orientation_key not in exif:
            return pil_image

        orientation = exif[orientation_key]
        transforms = {
            3: Image.Transpose.ROTATE_180,
            6: Image.Transpose.ROTATE_270,
            8: Image.Transpose.ROTATE_90,
        }
        if orientation in transforms:
            pil_image = pil_image.transpose(transforms[orientation])
    except Exception:
        pass
    return pil_image


def decode_and_downsample(b64_data: str, max_dim: int, filename: str = "") -> np.ndarray:
    """
    Decode a base64 image and IMMEDIATELY downsample it.
    
    This is the critical memory optimization:
    1. Decode base64 → raw bytes (transient)
    2. Open with PIL (transient, lazy-loaded)
    3. Fix EXIF orientation
    4. Downsample in PIL BEFORE converting to NumPy
    5. Convert to OpenCV BGR array (final, small)
    
    At no point do we hold a full-resolution NumPy array.
    """
    # Step 1: Decode base64 string to raw bytes
    if "," in b64_data and b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_data)
    
    # Step 2: Open with PIL (lazy — doesn't load full pixels yet)
    pil_img = Image.open(io.BytesIO(img_bytes))
    
    # Free the raw bytes immediately
    del img_bytes

    # Step 3: Fix EXIF orientation
    pil_img = fix_exif_orientation(pil_img)

    # Step 4: Downsample IN PIL (before creating the large NumPy array)
    orig_w, orig_h = pil_img.size
    scale = min(max_dim / max(orig_w, orig_h), 1.0)

    if scale < 1.0:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        logger.info(f"  {filename}: {orig_w}x{orig_h} → {new_w}x{new_h} (scale {scale:.2f})")
    else:
        logger.info(f"  {filename}: {orig_w}x{orig_h} (no resize needed)")

    # Step 5: Convert to RGB if needed
    if pil_img.mode == "RGBA":
        background = Image.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Step 6: Convert to OpenCV BGR (this is the only large array we keep)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Free PIL image immediately
    del pil_img

    return cv_img


def encode_to_jpeg_base64(img: np.ndarray, quality: int = 85) -> str:
    """Encode an OpenCV image to JPEG base64 string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(".jpg", img, encode_params)
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    del buffer
    return b64


def generate_thumbnail(img: np.ndarray, target_width: int = 480) -> np.ndarray:
    """Generate a thumbnail from the panorama."""
    h, w = img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)


def estimate_confidence(result: np.ndarray, source_count: int) -> float:
    """
    Estimate stitching confidence based on output quality metrics.
    Does NOT hold references to source images — only uses the result.
    """
    h, w = result.shape[:2]
    aspect = w / h if h > 0 else 0

    # Aspect ratio score (2:1 = perfect equirectangular)
    aspect_score = max(0, 1.0 - abs(aspect - 2.0) * 0.5)

    # Black border detection
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size
    black_pixels = np.sum(gray < 10)
    black_ratio = black_pixels / total_pixels if total_pixels > 0 else 0
    border_score = max(0, 1.0 - black_ratio * 5)

    # Edge density (well-stitched panoramas have consistent detail)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / total_pixels if total_pixels > 0 else 0
    edge_score = min(1.0, edge_density / 0.05) if edge_density > 0.005 else 0.3

    # Free intermediate arrays
    del gray, edges

    # Source coverage heuristic (more source images usually means wider coverage)
    coverage_score = min(1.0, source_count / 6.0) if source_count >= 2 else 0.3

    confidence = (
        aspect_score * 0.30 +
        border_score * 0.30 +
        edge_score * 0.20 +
        coverage_score * 0.20
    )

    return round(min(1.0, max(0.0, confidence)), 3)


# ─── Stitch Pipeline ───

def stitch_images(images: list[np.ndarray]) -> tuple[bool, np.ndarray | None, list[str]]:
    """
    Run OpenCV Stitcher on a list of pre-downsampled images.
    Returns (success, result_image, warnings).
    """
    warnings = []
    n = len(images)

    # Choose initial mode
    if n <= 3:
        primary_mode = cv2.Stitcher_SCANS
        fallback_mode = cv2.Stitcher_PANORAMA
        warnings.append("Using SCANS mode for small image set")
    else:
        primary_mode = cv2.Stitcher_PANORAMA
        fallback_mode = cv2.Stitcher_SCANS

    # Log memory estimate
    total_pixels = sum(img.shape[0] * img.shape[1] for img in images)
    est_mb = (total_pixels * 3) / (1024 * 1024)  # BGR = 3 bytes per pixel
    logger.info(f"Source images: {n}, total ~{est_mb:.0f} MB in RAM")
    logger.info(f"Running OpenCV Stitcher (primary mode)...")

    start = time.time()
    stitcher = cv2.Stitcher.create(primary_mode)
    status, result = stitcher.stitch(images)
    elapsed = time.time() - start

    status_messages = {
        cv2.Stitcher_OK: "OK",
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images with sufficient overlap",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed — images may not overlap enough",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed",
    }

    logger.info(f"Primary stitch: status={status_messages.get(status, status)}, {elapsed:.1f}s")

    if status == cv2.Stitcher_OK:
        del stitcher
        gc.collect()
        return True, result, warnings

    # Primary failed — try fallback mode
    primary_error = status_messages.get(status, f"Unknown error (code {status})")
    logger.info(f"Primary failed: {primary_error}. Trying fallback mode...")

    del stitcher, result
    gc.collect()

    stitcher_fb = cv2.Stitcher.create(fallback_mode)
    status2, result2 = stitcher_fb.stitch(images)

    logger.info(f"Fallback stitch: status={status_messages.get(status2, status2)}")

    del stitcher_fb
    gc.collect()

    if status2 == cv2.Stitcher_OK:
        warnings.append(f"Primary stitch failed ({primary_error}), used fallback mode")
        return True, result2, warnings
    else:
        fallback_error = status_messages.get(status2, f"Unknown error (code {status2})")
        return False, None, [primary_error, f"Fallback also failed: {fallback_error}"]


# ═══════════════════════════════════════════════════
#  AI RECONSTRUCT PIPELINE
# ═══════════════════════════════════════════════════

def detect_coverage_mask(panorama: np.ndarray) -> np.ndarray:
    """
    Detect which areas of the panorama have real image data vs empty/black regions.
    Returns a binary mask: 255 = real data, 0 = missing/empty.
    """
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    # Threshold: pixels below 12 are considered black/empty
    _, mask = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    # Morphological cleanup to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    del gray
    return mask


def estimate_room_geometry(panorama: np.ndarray, coverage_mask: np.ndarray) -> dict:
    """
    Estimate dominant colors, edge structure, and geometric properties
    of the covered regions. Used to guide inpainting of missing areas.
    """
    h, w = panorama.shape[:2]
    covered = panorama[coverage_mask > 0]

    if len(covered) == 0:
        return {"dominant_colors": [(128, 128, 128)], "mean_brightness": 128, "edge_density": 0.0}

    # Dominant colors via k-means on a sample
    sample_size = min(5000, len(covered))
    indices = np.random.choice(len(covered), sample_size, replace=False)
    sample = covered[indices].astype(np.float32)

    k = min(3, len(sample) // 10 + 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(sample, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    dominant_colors = [tuple(c.astype(int)) for c in centers]

    del sample, labels, centers

    # Mean brightness
    gray_covered = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray_covered[coverage_mask > 0]))

    # Edge density in covered area
    edges = cv2.Canny(gray_covered, 50, 150)
    covered_pixels = np.sum(coverage_mask > 0)
    edge_pixels_in_covered = np.sum((edges > 0) & (coverage_mask > 0))
    edge_density = edge_pixels_in_covered / covered_pixels if covered_pixels > 0 else 0.0

    del gray_covered, edges

    return {
        "dominant_colors": dominant_colors,
        "mean_brightness": mean_brightness,
        "edge_density": float(edge_density)
    }


def geometric_inpaint(panorama: np.ndarray, coverage_mask: np.ndarray, geometry: dict) -> np.ndarray:
    """
    Fill missing regions using geometric-aware inpainting.
    Strategy:
    1. Use OpenCV's Navier-Stokes inpainting for border regions (extends edges naturally)
    2. For large interior gaps, blend dominant room colors with noise for texture
    3. All operations preserve original pixels — only missing areas are filled

    Construction accuracy > visual perfection.
    """
    h, w = panorama.shape[:2]
    result = panorama.copy()

    # Inversion: the mask for inpainting = areas where coverage is 0
    inpaint_mask = cv2.bitwise_not(coverage_mask)

    missing_pixels = np.sum(inpaint_mask > 0)
    total_pixels = h * w
    missing_ratio = missing_pixels / total_pixels if total_pixels > 0 else 0

    if missing_pixels == 0:
        return result  # Nothing to fill

    logger.info(f"  AI Reconstruct: {missing_ratio*100:.1f}% missing ({missing_pixels} px)")

    # Step 1: Use OpenCV inpainting for regions near existing content
    # Telea method works well for extending existing geometry
    # Use a moderate radius to maintain construction detail
    inpaint_radius = max(3, min(12, int(min(h, w) * 0.01)))

    try:
        result = cv2.inpaint(result, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
        logger.info(f"  AI Reconstruct: Telea inpainting applied (radius={inpaint_radius})")
    except Exception as e:
        logger.warning(f"  AI Reconstruct: Telea inpainting failed ({e}), falling back to NS")
        try:
            result = cv2.inpaint(result, inpaint_mask, inpaint_radius, cv2.INPAINT_NS)
            logger.info(f"  AI Reconstruct: NS inpainting applied (radius={inpaint_radius})")
        except Exception as e2:
            logger.warning(f"  AI Reconstruct: NS inpainting also failed ({e2}), using color fill")
            # Fallback: fill with dominant color + noise for texture
            dom_color = geometry["dominant_colors"][0] if geometry["dominant_colors"] else (128, 128, 128)
            base_fill = np.full_like(panorama, dom_color, dtype=np.uint8)
            # Add slight noise for texture
            noise = np.random.normal(0, 8, base_fill.shape).astype(np.int16)
            base_fill = np.clip(base_fill.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # Apply Gaussian blur for smoothness
            base_fill = cv2.GaussianBlur(base_fill, (15, 15), 0)
            # Only fill missing areas
            result[inpaint_mask > 0] = base_fill[inpaint_mask > 0]
            del base_fill, noise

    # Step 2: Blend edges between original and reconstructed regions
    # Create a soft transition zone
    dilated_edge = cv2.dilate(inpaint_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=2)
    eroded_edge = cv2.erode(inpaint_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    blend_zone = cv2.bitwise_and(dilated_edge, cv2.bitwise_not(eroded_edge))

    if np.sum(blend_zone > 0) > 0:
        # Gaussian blur the transition zone for smooth blending
        blurred = cv2.GaussianBlur(result, (21, 21), 0)
        alpha = cv2.GaussianBlur(blend_zone.astype(np.float32) / 255.0, (21, 21), 0)
        alpha = np.stack([alpha] * 3, axis=-1)
        result = (result.astype(np.float32) * (1 - alpha * 0.3) + blurred.astype(np.float32) * (alpha * 0.3)).astype(np.uint8)
        del blurred, alpha

    del dilated_edge, eroded_edge, blend_zone

    # Step 3: Ensure original pixels are perfectly preserved
    result[coverage_mask > 0] = panorama[coverage_mask > 0]

    return result


def pad_to_equirectangular(panorama: np.ndarray, coverage_mask: np.ndarray, geometry: dict) -> tuple:
    """
    If the panorama is not 2:1 aspect ratio, pad it to equirectangular proportions.
    Uses geometric inpainting for the padded regions.
    Returns (padded_panorama, padded_mask).
    """
    h, w = panorama.shape[:2]
    aspect = w / h if h > 0 else 0

    # Already close to 2:1
    if 1.8 <= aspect <= 2.2:
        return panorama, coverage_mask

    # Target: 2:1 aspect ratio
    if aspect < 2.0:
        # Too tall or not wide enough — extend width
        target_w = int(h * 2.0)
        pad_total = target_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        padded = cv2.copyMakeBorder(panorama, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded_mask = cv2.copyMakeBorder(coverage_mask, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    else:
        # Too wide — extend height
        target_h = int(w / 2.0)
        pad_total = target_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        padded = cv2.copyMakeBorder(panorama, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded_mask = cv2.copyMakeBorder(coverage_mask, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

    logger.info(f"  AI Reconstruct: Padded {w}x{h} → {padded.shape[1]}x{padded.shape[0]} (target 2:1)")
    return padded, padded_mask


def run_ai_reconstruct(
    images: list[np.ndarray],
    room_id: str,
    room_name: str,
    confidence_threshold: float = 0.4
) -> dict:
    """
    Full AI reconstruction pipeline:
    1. Attempt OpenCV stitch
    2. Analyze coverage
    3. Estimate room geometry
    4. Pad to equirectangular if needed
    5. Inpaint missing regions
    6. Blend and preserve originals
    7. Return result with coverage metadata

    Returns dict with all fields needed for StitchResponse.
    """
    warnings = []
    start_time = time.time()

    # ─── Step 1: Attempt OpenCV stitch ───
    logger.info(f"  AI Reconstruct Step 1: OpenCV stitch attempt…")
    stitch_success, stitch_result, stitch_warnings = stitch_images(images)
    warnings.extend(stitch_warnings)

    source_count = len(images)

    if not stitch_success or stitch_result is None:
        # Complete stitch failure — cannot reconstruct without any stitched base
        logger.info(f"  AI Reconstruct: Stitch failed completely — cannot reconstruct")
        return {
            "success": False,
            "error": "OpenCV stitch failed completely. AI reconstruction requires at least a partial stitch. Images may lack sufficient overlap.",
            "warnings": warnings,
            "processingTimeMs": int((time.time() - start_time) * 1000)
        }

    # ─── Step 2: Analyze coverage ───
    logger.info(f"  AI Reconstruct Step 2: Analyzing coverage…")
    coverage_mask = detect_coverage_mask(stitch_result)
    total_pixels = coverage_mask.shape[0] * coverage_mask.shape[1]
    covered_pixels = int(np.sum(coverage_mask > 0))
    original_coverage = (covered_pixels / total_pixels * 100) if total_pixels > 0 else 0

    logger.info(f"  AI Reconstruct: Original coverage = {original_coverage:.1f}%")

    # Check stitch confidence
    confidence = estimate_confidence(stitch_result, source_count)
    logger.info(f"  AI Reconstruct: Stitch confidence = {confidence:.3f}")

    # If stitch is already good enough, skip reconstruction
    result_h, result_w = stitch_result.shape[:2]
    aspect = result_w / result_h if result_h > 0 else 0
    is_equirectangular = 1.5 <= aspect <= 2.5

    if confidence >= confidence_threshold and original_coverage >= 95:
        logger.info(f"  AI Reconstruct: Coverage ≥95% and confidence OK — no reconstruction needed")
        return {
            "success": True,
            "panorama": stitch_result,
            "coverage_mask": coverage_mask,
            "confidence": confidence,
            "original_coverage": original_coverage,
            "ai_generated_percent": 0.0,
            "ai_reconstructed": False,
            "reconstruction_method": "none_needed",
            "warnings": warnings,
            "processingTimeMs": int((time.time() - start_time) * 1000)
        }

    # ─── Step 3: Estimate room geometry ───
    logger.info(f"  AI Reconstruct Step 3: Estimating room geometry…")
    geometry = estimate_room_geometry(stitch_result, coverage_mask)

    # ─── Step 4: Pad to equirectangular ───
    logger.info(f"  AI Reconstruct Step 4: Aspect correction…")
    padded, padded_mask = pad_to_equirectangular(stitch_result, coverage_mask, geometry)

    # Free the original stitch result if padded is a new array
    if padded is not stitch_result:
        del stitch_result
        gc.collect()

    # ─── Step 5: Geometric inpainting ───
    logger.info(f"  AI Reconstruct Step 5: Geometric inpainting…")
    reconstructed = geometric_inpaint(padded, padded_mask, geometry)

    # Free the padded version
    del padded
    gc.collect()

    # ─── Step 6: Final coverage analysis ───
    final_h, final_w = reconstructed.shape[:2]
    final_total = final_h * final_w
    original_pixel_count = int(np.sum(padded_mask > 0))
    ai_pixel_count = final_total - original_pixel_count
    ai_generated_percent = (ai_pixel_count / final_total * 100) if final_total > 0 else 0
    final_original_coverage = (original_pixel_count / final_total * 100) if final_total > 0 else 0

    # Re-estimate confidence on the reconstructed result
    final_confidence = estimate_confidence(reconstructed, source_count)

    # Adjust confidence: penalize slightly for heavy AI fill
    if ai_generated_percent > 30:
        penalty = (ai_generated_percent - 30) * 0.005  # 0.5% per percent over 30%
        final_confidence = max(0.1, final_confidence - penalty)

    logger.info(f"  AI Reconstruct: Final dims {final_w}x{final_h}")
    logger.info(f"  AI Reconstruct: Original coverage {final_original_coverage:.1f}%, AI fill {ai_generated_percent:.1f}%")
    logger.info(f"  AI Reconstruct: Final confidence {final_confidence:.3f}")

    warnings.append(f"AI reconstruction assisted — {ai_generated_percent:.1f}% synthesized, {final_original_coverage:.1f}% original photography")

    return {
        "success": True,
        "panorama": reconstructed,
        "coverage_mask": padded_mask,
        "confidence": round(final_confidence, 3),
        "original_coverage": round(final_original_coverage, 1),
        "ai_generated_percent": round(ai_generated_percent, 1),
        "ai_reconstructed": True,
        "reconstruction_method": "geometric_inpaint",
        "warnings": warnings,
        "processingTimeMs": int((time.time() - start_time) * 1000)
    }


# ─── Endpoints ───

@app.get("/")
async def root():
    return {"service": "omma-stitch-service", "version": "1.2.0"}

@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "omma-stitch-service",
        "engine": "opencv",
        "version": "1.2.0",
        "opencv_version": cv2.__version__,
        "max_images": MAX_IMAGES,
        "memory_optimized": True,
        "max_dimension_few": MAX_DIM_FEW,
        "max_dimension_medium": MAX_DIM_MEDIUM,
        "max_dimension_many": MAX_DIM_MANY,
        "auth_required": bool(API_KEY),
        "capabilities": ["stitch", "ai_reconstruct"],
        "ai_reconstruct_available": True
    }

@app.get("/api/health")
async def api_health():
    return await health()


@app.post("/api/stitch")
async def stitch(request: StitchRequest):
    start_time = time.time()
    room_id = request.roomId
    room_name = request.roomName or room_id
    warnings = []

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"Stitch request: room='{room_name}' ({room_id}), {len(request.images)} images")
    logger.info(f"{'='*60}")

    # ─── Validation ───

    if len(request.images) < 2:
        return StitchResponse(
            success=False, isTrue360=False, roomId=room_id,
            error="At least 2 images are required for stitching",
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    if len(request.images) > MAX_IMAGES:
        return StitchResponse(
            success=False, isTrue360=False, roomId=room_id,
            error=f"Too many images ({len(request.images)}). Maximum is {MAX_IMAGES}.",
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    # ─── Determine max dimension based on image count ───
    max_dim = get_max_dim(len(request.images))
    logger.info(f"Memory budget: {len(request.images)} images → max dimension {max_dim}px")

    # ─── Decode + Downsample Images One at a Time ───
    # Critical: each image is decoded, downsampled, and the original freed
    # before the next one is decoded.

    cv_images = []
    decode_errors = []

    for img_input in request.images:
        try:
            # decode_and_downsample handles the full pipeline:
            # base64 → PIL → EXIF fix → resize in PIL → small OpenCV array
            cv_img = decode_and_downsample(img_input.data, max_dim, img_input.filename)
            cv_images.append(cv_img)
        except Exception as e:
            decode_errors.append(f"Image {img_input.index} ({img_input.filename}): {str(e)}")
            logger.error(f"  Failed to decode image {img_input.index}: {e}")

        # Clear the base64 data from the request object to free string memory
        img_input.data = ""

    # Force garbage collection after all decoding
    gc.collect()

    if decode_errors:
        warnings.extend([f"Decode warning: {e}" for e in decode_errors])

    if len(cv_images) < 2:
        return StitchResponse(
            success=False, isTrue360=False, roomId=room_id,
            error=f"Only {len(cv_images)} images decoded successfully. Need at least 2.",
            warnings=warnings,
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    # Log memory estimate for source images
    total_source_mb = sum(img.nbytes for img in cv_images) / (1024 * 1024)
    logger.info(f"All images decoded. {len(cv_images)} images, {total_source_mb:.0f} MB total in RAM")

    # ─── Run Stitcher ───

    try:
        success, result, stitch_warnings = stitch_images(cv_images)
        warnings.extend(stitch_warnings)
    except Exception as e:
        logger.error(f"Stitcher crashed: {e}")
        del cv_images
        gc.collect()
        return StitchResponse(
            success=False, isTrue360=False, roomId=room_id,
            error=f"OpenCV stitcher encountered an error: {str(e)}",
            warnings=warnings,
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    # Free source images immediately after stitching
    del cv_images
    gc.collect()

    if not success or result is None:
        return StitchResponse(
            success=False, isTrue360=False, isPreviewReady=False, roomId=room_id,
            error="Unable to create stable panorama. Images may lack sufficient overlap or feature points.",
            warnings=warnings,
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    # ─── Quality Assessment ───
    # Construction photography: images cannot be reshot.
    # Hard gates: confidence >= 0.4, width >= 800, height >= 400.
    # Equirectangular (2:1 aspect) is informational — it determines isTrue360 vs isPreviewReady,
    # but does NOT cause rejection. A valid stitch that isn't equirectangular is still accepted.

    result_h, result_w = result.shape[:2]
    logger.info(f"Stitch result: {result_w}x{result_h}")

    confidence = estimate_confidence(result, len(request.images))
    logger.info(f"Confidence: {confidence}")

    aspect_ratio = result_w / result_h if result_h > 0 else 0

    # Equirectangular check — informational only
    is_equirectangular = 1.5 <= aspect_ratio <= 2.5

    # Hard quality gates
    MIN_CONFIDENCE = 0.4
    MIN_WIDTH = 800
    MIN_HEIGHT = 400

    hard_gates_passed = (
        confidence >= MIN_CONFIDENCE and
        result_w >= MIN_WIDTH and
        result_h >= MIN_HEIGHT
    )

    # isTrue360 requires hard gates AND equirectangular projection
    is_true_360 = hard_gates_passed and is_equirectangular

    # isPreviewReady: hard gates passed, useful panorama, but not full equirectangular sphere
    is_preview_ready = hard_gates_passed and not is_equirectangular

    if not is_equirectangular:
        warnings.append(
            f"Standard panorama generated (not full 360 sphere). "
            f"Aspect ratio {aspect_ratio:.2f} — panorama viewer active."
        )

    if not hard_gates_passed:
        reasons = []
        if confidence < MIN_CONFIDENCE:
            reasons.append(f"confidence {confidence:.2f} < {MIN_CONFIDENCE}")
        if result_w < MIN_WIDTH:
            reasons.append(f"width {result_w} < {MIN_WIDTH}")
        if result_h < MIN_HEIGHT:
            reasons.append(f"height {result_h} < {MIN_HEIGHT}")
        warnings.append(f"Quality gates failed: {', '.join(reasons)}")

    logger.info(
        f"Quality: confidence={confidence}, equirect={is_equirectangular}, "
        f"hard_gates={'PASS' if hard_gates_passed else 'FAIL'}, "
        f"isTrue360={is_true_360}, isPreviewReady={is_preview_ready}"
    )

    # ─── Encode Output ───

    jpeg_quality = min(int(request.options.outputQuality * 100), JPEG_QUALITY)

    try:
        # Generate thumbnail first (smaller, less memory)
        thumb = generate_thumbnail(result, THUMBNAIL_WIDTH)
        thumbnail_b64 = encode_to_jpeg_base64(thumb, 75)
        del thumb
        gc.collect()

        # Encode full panorama
        panorama_b64 = encode_to_jpeg_base64(result, jpeg_quality)

        # Free the result array
        del result
        gc.collect()

    except Exception as e:
        del result
        gc.collect()
        return StitchResponse(
            success=False, isTrue360=False, roomId=room_id,
            error=f"Failed to encode panorama: {str(e)}",
            warnings=warnings,
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.info(f"{'='*60}")
    logger.info(
        f"DONE: {result_w}x{result_h}, confidence={confidence}, "
        f"isTrue360={is_true_360}, isPreviewReady={is_preview_ready}, "
        f"aspect={aspect_ratio:.2f}, {elapsed_ms}ms"
    )
    logger.info(f"{'='*60}")

    return StitchResponse(
        success=True,
        isTrue360=is_true_360,
        isPreviewReady=is_preview_ready,
        engine="opencv",
        roomId=room_id,
        panoramaBase64=panorama_b64,
        thumbnailBase64=thumbnail_b64,
        width=result_w,
        height=result_h,
        confidence=confidence,
        aspectRatio=round(aspect_ratio, 3),
        isEquirectangular=is_equirectangular,
        warnings=warnings,
        processingTimeMs=elapsed_ms
    )


# ═══════════════════════════════════════════════════
#  POST /api/ai-reconstruct — AI-Assisted Reconstruction
# ═══════════════════════════════════════════════════

@app.post("/api/ai-reconstruct")
async def ai_reconstruct(request: StitchRequest):
    start_time = time.time()
    room_id = request.roomId
    room_name = request.roomName or room_id
    warnings = []

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"AI RECONSTRUCT request: room='{room_name}' ({room_id}), {len(request.images)} images")
    logger.info(f"{'='*60}")

    # ─── Validation ───
    if len(request.images) < 2:
        return StitchResponse(
            success=False, isTrue360=False, engine="ai_reconstruct", roomId=room_id,
            error="At least 2 images are required",
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    if len(request.images) > MAX_IMAGES:
        return StitchResponse(
            success=False, isTrue360=False, engine="ai_reconstruct", roomId=room_id,
            error=f"Too many images ({len(request.images)}). Maximum is {MAX_IMAGES}.",
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    # ─── Decode + Downsample ───
    max_dim = get_max_dim(len(request.images))
    logger.info(f"Memory budget: {len(request.images)} images → max dimension {max_dim}px")

    cv_images = []
    decode_errors = []

    for img_input in request.images:
        try:
            cv_img = decode_and_downsample(img_input.data, max_dim, img_input.filename)
            cv_images.append(cv_img)
        except Exception as e:
            decode_errors.append(f"Image {img_input.index} ({img_input.filename}): {str(e)}")
            logger.error(f"  Failed to decode image {img_input.index}: {e}")
        img_input.data = ""

    gc.collect()

    if decode_errors:
        warnings.extend([f"Decode warning: {e}" for e in decode_errors])

    if len(cv_images) < 2:
        return StitchResponse(
            success=False, isTrue360=False, engine="ai_reconstruct", roomId=room_id,
            error=f"Only {len(cv_images)} images decoded successfully. Need at least 2.",
            warnings=warnings,
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    # ─── Run AI Reconstruct Pipeline ───
    try:
        result = run_ai_reconstruct(
            cv_images, room_id, room_name,
            confidence_threshold=0.4
        )
        warnings.extend(result.get("warnings", []))
    except Exception as e:
        logger.error(f"AI Reconstruct crashed: {e}")
        del cv_images
        gc.collect()
        return StitchResponse(
            success=False, isTrue360=False, engine="ai_reconstruct", roomId=room_id,
            error=f"AI reconstruction error: {str(e)}",
            warnings=warnings,
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    # Free source images
    del cv_images
    gc.collect()

    if not result.get("success"):
        return StitchResponse(
            success=False, isTrue360=False, isPreviewReady=False,
            engine="ai_reconstruct", roomId=room_id,
            error=result.get("error", "AI reconstruction failed"),
            warnings=warnings,
            aiReconstructed=False,
            processingTimeMs=result.get("processingTimeMs", int((time.time() - start_time) * 1000))
        )

    # ─── Encode output ───
    panorama = result["panorama"]
    coverage_mask = result["coverage_mask"]
    confidence = result["confidence"]
    ai_reconstructed = result.get("ai_reconstructed", False)

    result_h, result_w = panorama.shape[:2]
    aspect_ratio = result_w / result_h if result_h > 0 else 0
    is_equirectangular = 1.5 <= aspect_ratio <= 2.5

    # Quality assessment
    MIN_CONFIDENCE = 0.4
    MIN_WIDTH = 800
    MIN_HEIGHT = 400

    hard_gates_passed = (
        confidence >= MIN_CONFIDENCE and
        result_w >= MIN_WIDTH and
        result_h >= MIN_HEIGHT
    )

    is_true_360 = hard_gates_passed and is_equirectangular
    is_preview_ready = hard_gates_passed and not is_equirectangular

    jpeg_quality = min(int(request.options.outputQuality * 100), JPEG_QUALITY)

    try:
        thumb = generate_thumbnail(panorama, THUMBNAIL_WIDTH)
        thumbnail_b64 = encode_to_jpeg_base64(thumb, 75)
        del thumb
        gc.collect()

        panorama_b64 = encode_to_jpeg_base64(panorama, jpeg_quality)

        # Encode coverage mask as low-quality JPEG for overlay
        coverage_mask_b64 = None
        if coverage_mask is not None and ai_reconstructed:
            # Resize mask to reasonable thumbnail size for transfer
            mask_h = min(400, coverage_mask.shape[0])
            mask_w = int(mask_h * (coverage_mask.shape[1] / coverage_mask.shape[0]))
            mask_resized = cv2.resize(coverage_mask, (mask_w, mask_h), interpolation=cv2.INTER_AREA)
            coverage_mask_b64 = encode_to_jpeg_base64(
                cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR), 60
            )
            del mask_resized

        del panorama, coverage_mask
        gc.collect()

    except Exception as e:
        del panorama, coverage_mask
        gc.collect()
        return StitchResponse(
            success=False, isTrue360=False, engine="ai_reconstruct", roomId=room_id,
            error=f"Failed to encode panorama: {str(e)}",
            warnings=warnings,
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    elapsed_ms = int((time.time() - start_time) * 1000)

    logger.info(f"{'='*60}")
    logger.info(
        f"AI RECONSTRUCT DONE: {result_w}x{result_h}, confidence={confidence}, "
        f"ai_fill={result.get('ai_generated_percent', 0):.1f}%, "
        f"isTrue360={is_true_360}, {elapsed_ms}ms"
    )
    logger.info(f"{'='*60}")

    return StitchResponse(
        success=True,
        isTrue360=is_true_360,
        isPreviewReady=is_preview_ready,
        engine="ai_reconstruct",
        roomId=room_id,
        panoramaBase64=panorama_b64,
        thumbnailBase64=thumbnail_b64,
        width=result_w,
        height=result_h,
        confidence=confidence,
        aspectRatio=round(aspect_ratio, 3),
        isEquirectangular=is_equirectangular,
        warnings=warnings,
        processingTimeMs=elapsed_ms,
        aiReconstructed=ai_reconstructed,
        originalCoveragePercent=result.get("original_coverage"),
        aiGeneratedPercent=result.get("ai_generated_percent"),
        coverageMaskBase64=coverage_mask_b64,
        reconstructionMethod=result.get("reconstruction_method")
    )


# ═══════════════════════════════════════════════════
#  POST /api/deploy-managed-backend
#  Server-side deployment: reads own files, compares
#  against GitHub, commits changed files, pushes,
#  waits for Render, verifies health + ai-reconstruct.
# ═══════════════════════════════════════════════════

class DeployRequest(BaseModel):
    githubToken: str
    repo: str = "joetechsupport/360view"
    branch: str = "main"
    verifyUrl: str = ""  # e.g. https://three60view-noz3.onrender.com
    commitMessage: str = "Auto-update stitch backend"

class DeployFileStatus(BaseModel):
    filename: str
    status: str  # "pushed" | "unchanged" | "error"
    sha: Optional[str] = None
    error: Optional[str] = None

class DeployResponse(BaseModel):
    success: bool
    fileStatuses: list[DeployFileStatus] = []
    pushedCount: int = 0
    unchangedCount: int = 0
    errorCount: int = 0
    commitMessage: str = ""
    healthVerified: bool = False
    aiReconstructVerified: bool = False
    capabilities: list[str] = []
    error: Optional[str] = None
    warnings: list[str] = []
    processingTimeMs: int = 0


import httpx  # We'll use httpx for async HTTP calls


def _get_managed_files() -> dict:
    """
    Read the 5 managed backend files.
    main.py and requirements.txt are read from disk (/app/).
    Dockerfile, render.yaml, README.md are embedded constants.
    """
    files = {}

    # Read main.py from disk (this file — /app/main.py)
    main_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    try:
        with open(main_py_path, "r", encoding="utf-8") as f:
            files["main.py"] = f.read()
    except Exception as e:
        logger.error(f"Cannot read main.py from {main_py_path}: {e}")
        files["main.py"] = None

    # Read requirements.txt from disk
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    try:
        with open(req_path, "r", encoding="utf-8") as f:
            files["requirements.txt"] = f.read()
    except Exception as e:
        logger.error(f"Cannot read requirements.txt from {req_path}: {e}")
        files["requirements.txt"] = None

    # Embedded files
    files["Dockerfile"] = EMBEDDED_DOCKERFILE
    files["render.yaml"] = EMBEDDED_RENDER_YAML
    files["README.md"] = EMBEDDED_README

    return files


async def _github_get_file_sha(client, repo: str, branch: str, filename: str, token: str) -> Optional[str]:
    """Get the SHA of a file in the GitHub repo (for update)."""
    url = f"https://api.github.com/repos/{repo}/contents/{filename}?ref={branch}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        resp = await client.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("sha"), data.get("content", "")
        elif resp.status_code == 404:
            return None, None
        elif resp.status_code in (401, 403):
            raise PermissionError(f"GitHub auth failed (HTTP {resp.status_code})")
        else:
            return None, None
    except PermissionError:
        raise
    except Exception:
        return None, None


async def _github_push_file(client, repo: str, branch: str, filename: str,
                            content: str, sha: Optional[str], token: str,
                            commit_msg: str) -> tuple:
    """Push a file to GitHub via Contents API. Returns (success, new_sha, error)."""
    url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    b64_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    body = {
        "message": commit_msg,
        "content": b64_content,
        "branch": branch
    }
    if sha:
        body["sha"] = sha

    try:
        resp = await client.put(url, headers=headers, json=body, timeout=15)
        if resp.status_code in (200, 201):
            data = resp.json()
            new_sha = data.get("content", {}).get("sha", "")
            return True, new_sha, None
        else:
            err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            err_msg = err_data.get("message", f"HTTP {resp.status_code}")
            return False, None, err_msg
    except Exception as e:
        return False, None, str(e)


def _content_matches(local_content: str, remote_b64: str) -> bool:
    """Check if local content matches the base64-encoded remote content."""
    if not remote_b64:
        return False
    try:
        # GitHub returns base64 with newlines
        clean_b64 = remote_b64.replace("\n", "")
        remote_content = base64.b64decode(clean_b64).decode("utf-8")
        return local_content.strip() == remote_content.strip()
    except Exception:
        return False


@app.post("/api/deploy-managed-backend")
async def deploy_managed_backend(request: DeployRequest):
    start_time = time.time()
    warnings = []

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"DEPLOY MANAGED BACKEND → {request.repo}/{request.branch}")
    logger.info(f"{'='*60}")

    # Step 1: Read managed files
    managed_files = _get_managed_files()
    missing = [k for k, v in managed_files.items() if v is None]
    if missing:
        return DeployResponse(
            success=False,
            error=f"Cannot read managed files from disk: {', '.join(missing)}",
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    logger.info(f"  Read {len(managed_files)} managed files")

    # Step 2: Compare against GitHub and push changed files
    file_statuses = []
    pushed_count = 0
    unchanged_count = 0
    error_count = 0

    async with httpx.AsyncClient() as client:
        for filename, content in managed_files.items():
            try:
                # Get current SHA and content from GitHub
                result = await _github_get_file_sha(client, request.repo, request.branch, filename, request.githubToken)
                current_sha, current_content = result

                # Compare content
                if current_content and _content_matches(content, current_content):
                    logger.info(f"  {filename}: unchanged — skipping")
                    file_statuses.append(DeployFileStatus(
                        filename=filename, status="unchanged", sha=current_sha
                    ))
                    unchanged_count += 1
                    continue

                # Push changed file
                logger.info(f"  {filename}: changed — pushing…")
                success, new_sha, err = await _github_push_file(
                    client, request.repo, request.branch, filename,
                    content, current_sha, request.githubToken, request.commitMessage
                )

                if success:
                    logger.info(f"  {filename}: ✓ pushed (sha={new_sha[:8] if new_sha else '?'})")
                    file_statuses.append(DeployFileStatus(
                        filename=filename, status="pushed", sha=new_sha
                    ))
                    pushed_count += 1
                else:
                    logger.error(f"  {filename}: ✗ push failed — {err}")
                    file_statuses.append(DeployFileStatus(
                        filename=filename, status="error", error=err
                    ))
                    error_count += 1

            except PermissionError as e:
                return DeployResponse(
                    success=False,
                    error=str(e),
                    fileStatuses=file_statuses,
                    processingTimeMs=int((time.time() - start_time) * 1000)
                )
            except Exception as e:
                logger.error(f"  {filename}: ✗ error — {e}")
                file_statuses.append(DeployFileStatus(
                    filename=filename, status="error", error=str(e)
                ))
                error_count += 1

    if error_count > 0:
        return DeployResponse(
            success=False,
            error=f"{error_count} file(s) failed to push",
            fileStatuses=file_statuses,
            pushedCount=pushed_count,
            unchangedCount=unchanged_count,
            errorCount=error_count,
            commitMessage=request.commitMessage,
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    if pushed_count == 0:
        logger.info(f"  All files unchanged — no push needed")
        # Still verify endpoints if verifyUrl provided
        warnings.append("All files already up-to-date on GitHub — no push needed")

    # Step 3: Wait for Render deploy and verify
    health_verified = False
    ai_verified = False
    capabilities = []

    if request.verifyUrl:
        verify_base = request.verifyUrl.rstrip("/")
        logger.info(f"  Waiting for Render deploy…")

        # Wait before first check (give Render time to detect push)
        import asyncio
        if pushed_count > 0:
            await asyncio.sleep(5)

        # Poll for up to 5 minutes
        max_wait = 300
        poll_interval = 8
        elapsed = 0

        async with httpx.AsyncClient() as client:
            while elapsed < max_wait:
                try:
                    # Check /api/health
                    health_resp = await client.get(f"{verify_base}/api/health", timeout=10)
                    if health_resp.status_code == 200:
                        health_data = health_resp.json()
                        if health_data.get("ok") and health_data.get("engine") == "opencv":
                            capabilities = health_data.get("capabilities", [])
                            health_verified = True
                            logger.info(f"  Health verified: version={health_data.get('version')}, capabilities={capabilities}")

                            # Check /api/ai-reconstruct exists (OPTIONS or small POST)
                            if "ai_reconstruct" in capabilities:
                                try:
                                    # Send minimal request to verify route exists (will fail validation but route won't 404)
                                    ai_resp = await client.post(
                                        f"{verify_base}/api/ai-reconstruct",
                                        json={"roomId": "_verify", "images": []},
                                        timeout=10
                                    )
                                    # 422 = route exists but validation failed (expected)
                                    # 200/400 = route exists
                                    # 404 = route doesn't exist
                                    if ai_resp.status_code != 404:
                                        ai_verified = True
                                        logger.info(f"  ai-reconstruct route verified (HTTP {ai_resp.status_code})")
                                    else:
                                        logger.warning(f"  ai-reconstruct route returned 404 — old deploy still running")
                                        health_verified = False  # Force continued polling
                                except Exception as e:
                                    logger.warning(f"  ai-reconstruct check failed: {e}")

                            if health_verified:
                                break
                except Exception as e:
                    logger.info(f"  Health check attempt failed ({elapsed}s): {e}")

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

        if not health_verified:
            warnings.append(f"Deploy verification timed out after {max_wait}s. Render may still be deploying.")

    logger.info(f"{'='*60}")
    logger.info(f"DEPLOY COMPLETE: pushed={pushed_count}, unchanged={unchanged_count}, errors={error_count}")
    logger.info(f"  health_verified={health_verified}, ai_verified={ai_verified}")
    logger.info(f"{'='*60}")

    return DeployResponse(
        success=True,
        fileStatuses=file_statuses,
        pushedCount=pushed_count,
        unchangedCount=unchanged_count,
        errorCount=error_count,
        commitMessage=request.commitMessage,
        healthVerified=health_verified,
        aiReconstructVerified=ai_verified,
        capabilities=capabilities,
        warnings=warnings,
        processingTimeMs=int((time.time() - start_time) * 1000)
    )


# ─── Error Handlers ───

@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "success": False, "isTrue360": False, "engine": "opencv",
            "error": f"Invalid request format: {str(exc)}", "warnings": []
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False, "isTrue360": False, "engine": "opencv",
            "error": "Internal server error", "warnings": []
        }
    )


# ─── Run ───

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
