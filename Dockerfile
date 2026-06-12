FROM python:3.11-slim

# OpenCV headless system dependencies (minimal set)
# Note: libgl1-mesa-glx removed — no longer available on current Debian base.
# libgl1 provides the same OpenGL runtime needed by OpenCV headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Render provides PORT via environment
ENV PORT=8000

EXPOSE ${PORT}

CMD ["python", "main.py"]
