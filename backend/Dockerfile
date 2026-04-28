FROM python:3.11-slim

WORKDIR /app

# System deps for scipy, matplotlib, torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir matplotlib hi-ml-multimodal

# Copy the entire backend source
# Build context is RadGuard-AI-Engine/ (set in docker-compose.yml)
COPY . .

# Results folder for heatmap PNGs
RUN mkdir -p /workspace/results

# NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
