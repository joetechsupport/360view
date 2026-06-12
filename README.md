# OMMA Stitch Service

OpenCV panorama stitching backend for **OMMA 360 Builder**.

---

## Endpoints

| Method | Path         | Auth required | Description                    |
|--------|--------------|---------------|--------------------------------|
| GET    | /health      | No            | Health check                   |
| GET    | /api/health  | No            | Health check (alias)           |
| POST   | /api/stitch  | If key set    | Stitch images into a panorama  |

---

## Quick Start (local)

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
# Service is now at http://localhost:8000
```

Test the health endpoint:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "ok": true,
  "service": "omma-stitch-service",
  "engine": "opencv",
  "version": "1.0.0",
  "opencvVersion": "4.9.0",
  "maxImages": 40,
  "maxImageDimension": 3000,
  "confidenceThreshold": 0.70,
  "auth": false
}
```

---

## Deploy to Render

1. Push this folder to a GitHub repository.
2. Go to [render.com](https://render.com) → **New Web Service**.
3. Connect your repo.
4. Render auto-detects `render.yaml` — click **Apply**.
5. Optionally set `OMMA_API_KEY` in the Render dashboard.
6. Wait for the build to complete (first build takes 3–5 minutes).
7. Copy the service URL.

---

## Configure OMMA Frontend

1. Open OMMA 360 Builder → **Settings** tab.
2. Select engine: **opencv**.
3. Set stitch endpoint: `https://your-service.onrender.com/api/stitch`
4. Optionally set API key if `OMMA_API_KEY` is configured.
5. Click **Test Connection** → should show ✓ Connected.

---

## POST /api/stitch

### Request

```json
{
  "roomId": "room_abc123",
  "roomName": "Kitchen",
  "imageCount": 6,
  "images": [
    {
      "index": 0,
      "filename": "image-001.jpg",
      "data": "BASE64_ENCODED_JPEG",
      "mimeType": "image/jpeg"
    }
  ],
  "options": {
    "outputFormat": "jpeg",
    "outputQuality": 0.92,
    "targetProjection": "equirectangular"
  }
}
```

### Success Response

```json
{
  "success": true,
  "isTrue360": true,
  "engine": "opencv",
  "roomId": "room_abc123",
  "panoramaBase64": "BASE64_JPEG",
  "thumbnailBase64": "BASE64_JPEG",
  "width": 4096,
  "height": 2048,
  "aspectRatio": 2.0,
  "confidence": 0.94,
  "qualityGates": {
    "minDimensions": true,
    "aspectRatio": true,
    "borderArtifacts": true,
    "edgeDensity": true
  },
  "warnings": [],
  "processingTimeMs": 4821
}
```

### Failure Response

```json
{
  "success": false,
  "isTrue360": false,
  "engine": "opencv",
  "roomId": "room_abc123",
  "error": "Not enough overlapping features — add more images.",
  "warnings": [],
  "processingTimeMs": 1203
}
```

---

## Quality Gates

A room only receives `isTrue360: true` when ALL of the following pass:

| Gate              | Condition                                         |
|-------------------|---------------------------------------------------|
| `minDimensions`   | Output is at least 1024 × 256 px                  |
| `aspectRatio`     | Width / Height is between 1.5 and 3.0             |
| `borderArtifacts` | Black pixel fraction < 20 %                       |
| `edgeDensity`     | Edge pixel fraction > 1 % (image not blank)       |
| `confidence`      | Composite score ≥ threshold (default 0.70)        |

If any gate fails the room is returned with `isTrue360: false` and a
detailed `warnings` list explaining which gate failed.

---

## Environment Variables

| Variable              | Default | Description                                      |
|-----------------------|---------|--------------------------------------------------|
| `PORT`                | `8000`  | HTTP port                                        |
| `OMMA_API_KEY`        | `""`    | If set, require `Authorization: Bearer <key>`   |
| `CORS_ORIGINS`        | `"*"`   | Comma-separated allowed origins                  |
| `MAX_IMAGE_DIMENSION` | `3000`  | Resize any axis exceeding this value             |
| `MAX_IMAGES`          | `40`    | Hard cap on images per request                   |
| `CONFIDENCE_THRESHOLD`| `0.70`  | Minimum composite score to set `isTrue360: true` |

---

## Tips for Good Stitching Results

- Shoot **overlapping** photos — aim for 30–50 % overlap between adjacent frames.
- Keep the camera at a **fixed position** and rotate horizontally.
- Avoid subjects in motion between shots.
- Use consistent exposure settings — auto exposure can cause seam lines.
- 6–12 images per room is a good starting point for a full 360.
- Shoot at the same height for every frame in a room.

---

## Architecture

```
OMMA 360 Builder (frontend)
  ↓  POST /api/stitch (base64 images)
OMMA Stitch Service (this repo)
  ↓  cv2.Stitcher.stitch()
OpenCV panorama stitching
  ↓  quality gates
  ↓  encode JPEG
  ↑  panoramaBase64 + confidence
OMMA 360 Builder (frontend)
  ↓  isTrue360 = true → status = complete
  ↓  isTrue360 = false → status = needs_review
360 Viewer / Export
```

OMMA = Builder + Stitcher + Exporter  
Shared contract = ZIP package + tour.json schema

---

## License

MIT — OMMA 360 Builder internal service.
