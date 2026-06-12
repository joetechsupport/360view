"""
OMMA 360 Builder — OpenCV Stitch Service
=========================================
Production stitching backend for OMMA.
Receives overlapping room photos, runs OpenCV Stitcher,
returns a real equirectangular panorama or honest failure.

MEMORY-OPTIMIZED for Render free/starter (512 MB).
All images are immediately downsampled on decode.
No full-resolution copies are ever held in memory.
"""

import os
import io
import gc
import base64
import time
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
    isPreviewReady: bool = False
    engine: str = "opencv"
    roomId: str = ""
    panoramaBase64: Optional[str] = None
    thumbnailBase64: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    confidence: Optional[float] = None
    warnings: list[str] = []
    error: Optional[str] = None
    processingTimeMs: Optional[int] = None


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
        "auth_required": bool(API_KEY)
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
            success=False, isTrue360=False, roomId=room_id,
            error="Unable to create stable panorama. Images may lack sufficient overlap or feature points.",
            warnings=warnings,
            processingTimeMs=int((time.time() - start_time) * 1000)
        )

    # ─── Quality Assessment ───

    result_h, result_w = result.shape[:2]
    logger.info(f"Stitch result: {result_w}x{result_h}")

    confidence = estimate_confidence(result, len(request.images))
    logger.info(f"Confidence: {confidence}")

    aspect_ratio = result_w / result_h if result_h > 0 else 0
    is_equirectangular = 1.5 <= aspect_ratio <= 2.5

    # v1.2.0 construction mode:
    # Completed construction jobs often use existing project photos that cannot be re-shot.
    # A non-2:1 stitched output can still be a useful standard panorama preview.
    if not is_equirectangular:
        warnings.append(
            f"Standard panorama generated: aspect ratio {aspect_ratio:.2f} is not full 2:1 equirectangular."
        )

    min_dimensions_ok = result_w >= 800 and result_h >= 400
    is_preview_ready = (
        confidence >= 0.40 and
        len(request.images) >= 5 and
        min_dimensions_ok
    )

    is_true_360 = (
        confidence >= 0.70 and
        min_dimensions_ok and
        is_equirectangular
    )

    if not is_preview_ready:
        reasons = []
        if confidence < 0.40:
            reasons.append(f"confidence {confidence} < 0.40")
        if len(request.images) < 5:
            reasons.append(f"source images {len(request.images)} < 5")
        if result_w < 800:
            reasons.append(f"width {result_w} < 800")
        if result_h < 400:
            reasons.append(f"height {result_h} < 400")
        warnings.append(f"Not preview ready: {', '.join(reasons)}")
    elif not is_true_360:
        warnings.append(
            "Preview ready, but not marked true 360 because confidence is below 0.70 or output is not equirectangular."
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
    logger.info(f"DONE: {result_w}x{result_h}, confidence={confidence}, true360={is_true_360}, previewReady={is_preview_ready}, {elapsed_ms}ms")
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
        warnings=warnings,
        processingTimeMs=elapsed_ms
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
