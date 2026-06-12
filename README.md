# OMMA Stitch Service v1.2.0

FastAPI + OpenCV backend for OMMA 360 Builder.

## What changed in v1.2.0

- Keeps the memory-optimized decode/downsample pipeline.
- Fixes Docker dependency to use `libgl1` instead of `libgl1-mesa-glx`.
- Adds construction-friendly quality gates.
- Accepts usable stitched standard panoramas as `isPreviewReady: true`.
- Keeps `isTrue360: true` only for higher-confidence, equirectangular outputs.
- Treats non-2:1 aspect ratio as informational instead of automatic failure.

## Endpoints

- `GET /health`
- `GET /api/health`
- `POST /api/stitch`

## Key response fields

```json
{
  "success": true,
  "isTrue360": false,
  "isPreviewReady": true,
  "engine": "opencv",
  "confidence": 0.49,
  "warnings": [
    "Standard panorama generated: aspect ratio 1.48 is not full 2:1 equirectangular.",
    "Preview ready, but not marked true 360 because confidence is below 0.70 or output is not equirectangular."
  ]
}
```

## Quality behavior

### True 360

Requires:

- confidence >= 0.70
- width >= 800
- height >= 400
- aspect ratio between 1.5 and 2.5

### Preview Ready

Requires:

- confidence >= 0.40
- at least 5 source images
- width >= 800
- height >= 400

This mode is intended for completed construction projects where the photo set cannot be re-shot.

## Render deployment

Files must be at the repository root:

- `main.py`
- `requirements.txt`
- `Dockerfile`
- `render.yaml`
- `README.md`

Render should auto-deploy after commit.
