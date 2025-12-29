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

# Start from /app/api so imports like "from ai_engine" work
WORKDIR /app/api

# Start command
# Now we use "main:app" because we are inside the api directory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
