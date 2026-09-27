# ==============================================================================
# Bureau of Indian Standards (BIS) AI Compliance Assistant
# Production-Ready Docker Container
# ==============================================================================

FROM python:3.11-slim AS runtime

# Set environment flags for Python runtime in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    PYTHONPATH=/app/src:/app

# Install minimal OS dependencies for health checking and binary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user for container security
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Set working directory
WORKDIR /app

# Upgrade pip and install CPU-optimized PyTorch first for fast, lean image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch>=2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configuration (respecting .dockerignore)
COPY --chown=appuser:appgroup app.py .
COPY --chown=appuser:appgroup index.html .
COPY --chown=appuser:appgroup product_standard_map.json .
COPY --chown=appuser:appgroup labs_directory.json .
COPY --chown=appuser:appgroup schema.sql .
COPY --chown=appuser:appgroup sources.yaml .
COPY --chown=appuser:appgroup phase1_data_quality_report.json .
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup static/ ./static/
COPY --chown=appuser:appgroup scripts/ ./scripts/
COPY --chown=appuser:appgroup tests/ ./tests/
COPY --chown=appuser:appgroup processed_chunks.jsonl .
COPY --chown=appuser:appgroup vector_index/bm25_index.pkl ./vector_index/bm25_index.pkl
COPY --chown=appuser:appgroup .env.example .
COPY --chown=appuser:appgroup README.md .

# Ensure appuser owns /app directory for SQLite feedback logs and runtime files
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose HTTP API port
EXPOSE 8000

# Container healthcheck using FastAPI production /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Launch production server with uvicorn listening on all interfaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
