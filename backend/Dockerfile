FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .

# CPU-only torch keeps the image small; the hosts this runs on have no GPU
RUN python -m pip install --upgrade pip && \
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Include the embedding model so a restarted container does not download it again
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"

COPY app ./app

# Hugging Face Spaces runs the container as a non-root user
RUN mkdir -p /app/data /app/logs && chmod -R 777 /app/data /app/logs /app/.cache

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
