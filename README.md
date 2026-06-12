# OMMA Stitch Service — OpenCV Backend

Production stitching backend for the OMMA 360 Builder.  
**Memory-optimized for Render starter tier (512 MB).**

## Memory Strategy

The root cause of OOM crashes: holding full-resolution images in RAM.

**Fix:** Every image is downsampled *during decode*, before a full-resolution NumPy array ever exists.

```
base64 string → raw bytes → PIL (lazy) → EXIF fix → resize in PIL → small OpenCV array
```

The max dimension scales based on image count:

| Images | Max Dimension | ~RAM per image | 12 images total |
|--------|--------------|----------------|-----------------|
| ≤ 8    | 1600px       | ~5.5 MB        | ~66 MB          |
| 9–15   | 1200px       | ~4 MB          | ~48 MB          |
| > 15   | 800px        | ~2 MB          | ~40 MB          |

OpenCV Stitcher internally uses ~2-3× the source memory, so peak usage stays under ~300 MB — safely within the 512 MB container limit.

## Quick Start (Local)

```bash
cd stitch-service
pip install -r requirements.txt
python main.py
```

Service runs at `http://localhost:8000`.

## Quick Start (Docker)

```bash
cd stitch-service
docker build -t omma-stitch .
docker run -p 8000:8000 omma-stitch
```

## Deploy to Render

1. Push this `stitch-service/` directory to a GitHub repo
2. Create a new **Web Service** on Render
3. Point it to the repo — Render auto-detects the Dockerfile
4. Optionally set `OMMA_API_KEY` in the Render dashboard
5. After deploy, your endpoint is: `https://your-service.onrender.com/api/stitch`

## Connect to OMMA

1. Open OMMA 360 Builder
2. Go to **Settings** tab
3. Select **opencv** engine
4. Enter endpoint URL: `https://your-service.onrender.com/api/stitch`
5. Enter API key if configured
6. Click **Test Connection** — should show ✓ Connected
7. Go to any room with 2+ uploaded images
8. Click **Build 360 Room**

## API Reference

### GET /health  &  GET /api/health

```json
{
  "ok": true,
  "service": "omma-stitch-service",
  "engine": "opencv",
  "version": "1.1.0",
  "opencv_version": "4.9.0",
  "max_images": 30,
  "memory_optimized": true,
  "max_dimension_few": 1600,
  "max_dimension_medium": 1200,
  "max_dimension_many": 800,
  "auth_required": false
}
```

### POST /api/stitch

**Request:**
```json
{
  "roomId": "kitchen_001",
  "roomName": "Kitchen",
  "images": [
    {
      "index": 0,
      "filename": "IMG_001.jpg",
      "data": "BASE64_STRING",
      "mimeType": "image/jpeg"
    }
  ],
  "options": {
    "outputFormat": "jpeg",
    "outputQuality": 0.85,
    "targetProjection": "equirectangular"
  }
}
```

**Success:**
```json
{
  "success": true,
  "isTrue360": true,
  "engine": "opencv",
  "roomId": "kitchen_001",
  "panoramaBase64": "...",
  "thumbnailBase64": "...",
  "width": 3200,
  "height": 1600,
  "confidence": 0.91,
  "warnings": [],
  "processingTimeMs": 4200
}
```

**Failure:**
```json
{
  "success": false,
  "isTrue360": false,
  "engine": "opencv",
  "roomId": "kitchen_001",
  "error": "Unable to create stable panorama. Images may lack sufficient overlap or feature points.",
  "warnings": [],
  "processingTimeMs": 1500
}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | 8000 | Server port |
| `MAX_IMAGES` | 30 | Max images per request |
| `MAX_PAYLOAD_MB` | 100 | Max request body size |
| `JPEG_QUALITY` | 85 | Max output JPEG quality |
| `THUMBNAIL_WIDTH` | 480 | Thumbnail width in px |
| `OMMA_API_KEY` | "" | If set, requires Bearer token |

## Memory-related configs (hardcoded, tunable in code)

| Constant | Value | Description |
|---|---|---|
| `MAX_DIM_FEW` | 1600 | Max px when ≤ 8 images |
| `MAX_DIM_MEDIUM` | 1200 | Max px when 9–15 images |
| `MAX_DIM_MANY` | 800 | Max px when > 15 images |

## Verify Deployment

```bash
curl https://your-service.onrender.com/health
curl https://your-service.onrender.com/api/health
```

Both should return `{"ok": true, ...}`.

## End-to-End Flow

```
OMMA Frontend (browser)
  │
  ├── User uploads images to a room
  ├── User selects "opencv" engine
  ├── User enters service URL + tests connection
  ├── User clicks "Build 360 Room"
  │
  ├── POST /api/stitch ──────────► OMMA Stitch Service
  │    • base64 images                │
  │                                   ├── Decode each image
  │                                   ├── IMMEDIATELY downsample (≤1600px)
  │                                   ├── Free original bytes
  │                                   ├── Fix EXIF orientation
  │                                   ├── Run OpenCV Stitcher
  │                                   ├── Quality gates
  │                                   ├── Encode JPEG + thumbnail
  │                                   ├── Free all arrays
  │                                   │
  │   ◄── JSON response ─────────────┘
  │    • panoramaBase64 + thumbnailBase64
  │    • confidence, dimensions
  │    • isTrue360 = true/false
  │
  ├── Display in 360 viewer if isTrue360
  ├── Show diagnostic warning if not
  └── Export when ready
```
