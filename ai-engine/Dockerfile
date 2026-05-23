FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Clone Stanford CheXbert (public repo — only source code, no weights)
RUN git clone --depth 1 https://github.com/stanfordmlgroup/CheXbert.git /app/CheXbert

# Install CheXbert's own dependencies
RUN pip install --no-cache-dir scikit-learn tqdm || true

COPY inference/ ./inference/
COPY main.py .

RUN mkdir -p /tmp/results

ENV CHEXBERT_DIR=/app/CheXbert
ENV CHEXBERT_CKPT=/app/CheXbert/src/chexbert.pth

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
