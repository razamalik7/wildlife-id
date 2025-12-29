# Use Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to cache install
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
# This copies everything in /api to /app/api
COPY api ./api

# Copy Parks Data (Essential for WhereToSee)
# We recreate the folder structure so the relative path "../frontend/public/parks.json" still works
COPY frontend/public/parks.json ./frontend/public/parks.json

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Start command
# We run from root so "api.main" module path works
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
