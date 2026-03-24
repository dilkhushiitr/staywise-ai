# ─────────────────────────────────────────────────────────────
#  StayWise AI — Dockerfile
#  Multi-stage build: keeps final image lean
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────
COPY configs/   configs/
COPY src/       src/
COPY main.py    .

# ── Copy processed data (pipeline output) ────────────────────
# data/processed must exist before building image
# Run python3 main.py --step 1 locally first
COPY data/processed/  data/processed/

# ── Copy vector store ─────────────────────────────────────────
# data/vectorstore must exist before building image
# Run python3 main.py --step 2 locally first
COPY data/vectorstore/ data/vectorstore/

# ── Copy env file ─────────────────────────────────────────────
# Copy env.example which has the API key
COPY env.example env.example

# ── Logs directory ────────────────────────────────────────────
RUN mkdir -p logs

# ── Expose port ───────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start the API server ──────────────────────────────────────
CMD ["python3", "main.py", "--step", "5"]
