# AnimlVision API — HuggingFace Space (docker sdk, app_port 7860)
FROM python:3.10-slim

# Unbuffered output so container logs actually show startup progress —
# the April 2026 launch timeout was undebuggable because nothing flushed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# Dependencies first (cached layer)
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Model weights from the GitHub release (cached layer; not in git history)
ADD https://github.com/razamalik7/wildlife-id/releases/download/weights-v1/oogway_b3_best.pth /app/api/oogway_b3_best.pth
ADD https://github.com/razamalik7/wildlife-id/releases/download/weights-v1/oogway_convnext_final.pth /app/api/oogway_convnext_final.pth

# Runtime code + data only — NOT the training/scraping scripts
COPY api/main.py api/ai_engine.py api/species_config.json /app/api/
COPY frontend/public/parks.json /app/frontend/public/parks.json

WORKDIR /app/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
