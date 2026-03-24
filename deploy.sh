#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  StayWise AI — Deploy Script
#  Usage: bash deploy.sh
# ─────────────────────────────────────────────────────────────

set -e   # exit on any error

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   STAYWISE AI — DEPLOYMENT                      ║"
echo "╚══════════════════════════════════════════════════╝"

# ── Step 1: Check required files exist ───────────────────────
echo ""
echo "▶  Checking required files..."

if [ ! -f "data/processed/unified_hotels.csv" ]; then
    echo "❌  data/processed/unified_hotels.csv not found!"
    echo "    Run: python3 main.py --step 1"
    exit 1
fi

if [ ! -d "data/vectorstore" ]; then
    echo "❌  data/vectorstore not found!"
    echo "    Run: python3 main.py --step 2"
    exit 1
fi

if [ ! -f "env.example" ]; then
    echo "❌  env.example not found!"
    echo "    Create it with your GROQ_API_KEY"
    exit 1
fi

echo "   ✅  All required files found"

# ── Step 2: Check Docker is running ──────────────────────────
echo ""
echo "▶  Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "❌  Docker is not running. Start Docker Desktop first."
    exit 1
fi
echo "   ✅  Docker is running"

# ── Step 3: Build Docker image ────────────────────────────────
echo ""
echo "▶  Building Docker image (this takes 2-3 minutes first time)..."
docker-compose build

echo "   ✅  Image built"

# ── Step 4: Start container ───────────────────────────────────
echo ""
echo "▶  Starting StayWise AI container..."
docker-compose up -d

echo "   ✅  Container started"

# ── Step 5: Wait for health check ────────────────────────────
echo ""
echo "▶  Waiting for API to be ready..."
sleep 5

for i in {1..10}; do
    if curl -sf http://localhost:8000/health > /dev/null; then
        echo "   ✅  API is healthy!"
        break
    fi
    echo "   Waiting... ($i/10)"
    sleep 3
done

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   ✅  STAYWISE AI IS RUNNING                     ║"
echo "║                                                  ║"
echo "║   Swagger UI  → http://localhost:8000/docs       ║"
echo "║   Health      → http://localhost:8000/health     ║"
echo "║   API Base    → http://localhost:8000/api/v1     ║"
echo "║                                                  ║"
echo "║   To stop:  docker-compose down                  ║"
echo "║   To logs:  docker-compose logs -f               ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
