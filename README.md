# staywise-ai
AI-powered hotel decision intelligence system using RAG + LLM


# 🏨 StayWise AI — Hotel Decision Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker)
![RAG](https://img.shields.io/badge/RAG-ChromaDB-orange?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama3-purple?style=flat-square)

**An end-to-end AI-powered hotel recommendation system for India**
that aggregates multi-source hotel data, uses RAG + LLM for deep review
understanding, and provides personalized, explainable recommendations.

</div>

---

## 📌 Project Overview

StayWise AI solves the problem of **hotel booking overload** — hundreds of
reviews, inconsistent ratings across platforms, and no explanation behind
recommendations. It aggregates data from **MakeMyTrip, Goibibo, and Google**
across **560 Indian cities**, and uses a full AI stack to answer natural
language questions about hotels.

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
FastAPI (REST API)
    │
    ├── Ranking Engine ──── Composite Score + Dynamic Re-ranking
    │
    ├── RAG System ─────── ChromaDB Vector Search (9,557 chunks)
    │       │
    │       └── Sentence Transformers (all-MiniLM-L6-v2)
    │
    └── LLM Layer ──────── Groq (Llama 3) via Groq API
            │
            ├── Q&A          (grounded in real hotel data)
            ├── Summarizer   (pros / cons / verdict)
            └── Explainer    (why this hotel for YOU)
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Smart Search** | Top 20 hotels filtered by city, budget, stars |
| 🤖 **RAG Q&A** | Ask anything — answers grounded in real reviews |
| 📋 **Summarizer** | Auto-generated pros/cons for any hotel |
| 🎯 **Explainable Recommendations** | AI explains WHY each hotel is recommended |
| ⚡ **Dynamic Ranking** | "Best food" query boosts food scores; "Clean rooms" boosts cleanliness |
| 👨‍👩‍👧 **Personalization** | Re-ranks by travel type: family / couple / business / solo |
| ⚠️ **Hidden Issues Detection** | Flags hotels with recurring complaints |
| 🔄 **Multi-source Trust Score** | Unified 0–10 score from 3 platforms |
| 🐳 **Dockerized** | One-command deployment |

---

## 📊 Data

| Source | Hotels | Cities |
|--------|--------|--------|
| Goibibo | 3,995 | Pan India |
| Google Reviews | 1,004 | 51 cities |
| MakeMyTrip | 580 | 6 major cities |
| **Total** | **5,579** | **560** |

Price range: ₹404 – ₹1,25,000/night

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **ML / NLP** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector DB** | ChromaDB (9,557 embeddings, 384 dimensions) |
| **LLM** | Llama 3 via Groq API |
| **Data** | Pandas, NumPy |
| **Config** | YAML-based central configuration |
| **Deployment** | Docker, Docker Compose |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/staywise-ai.git
cd staywise-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add API key
```bash
# Open env.example and add your free Groq key
# Get it at: https://console.groq.com
GROQ_API_KEY=your-key-here
```

### 4. Run the full pipeline
```bash
python3 main.py --step 1   # Data Pipeline
python3 main.py --step 2   # RAG System (embeddings)
python3 main.py --step 3   # LLM Integration test
python3 main.py --step 4   # Ranking Engine test
python3 main.py --step 5   # Start API server
```

### 5. Or deploy with Docker
```bash
bash deploy.sh
```

Open **http://localhost:8000/docs** for interactive API docs.

---

## 📡 API Endpoints

```
POST /api/v1/search       → Top hotels by city + dynamic ranking
POST /api/v1/compare      → Side-by-side hotel comparison
POST /api/v1/personalize  → Hotels ranked by travel type
POST /api/v1/ask          → Natural language Q&A
POST /api/v1/summarize    → Pros/cons summary for any hotel
POST /api/v1/recommend    → Explainable AI recommendations
POST /api/v1/issues       → Hidden issue detection
GET  /health              → Health check
```

### Example: Ask a question
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Best hotels in Goa with a pool under ₹5000?", "city": "goa"}'
```

### Example: Get recommendations
```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "city": "mumbai",
    "travel_type": "business",
    "budget": 6000,
    "priorities": ["wifi", "city center"]
  }'
```

---

## 📁 Project Structure

```
staywise-ai/
├── configs/config.yaml        ← all settings
├── data/
│   ├── raw/                   ← source CSVs
│   └── processed/             ← pipeline outputs
├── src/
│   ├── ingestion/             ← data loading (CSV + Google Places API)
│   ├── processing/            ← cleaning, standardizing, merging
│   ├── rag/                   ← chunking, embeddings, ChromaDB
│   ├── llm/                   ← Q&A, summarizer, explainer
│   ├── ranking/               ← scorer, ranker, personalizer
│   └── api/                   ← FastAPI endpoints
├── Dockerfile
├── docker-compose.yml
├── main.py                    ← single entry point
└── requirements.txt
```

---

## 🔮 Future Scope

- [ ] Real-time Google Places API integration (config flag ready)
- [ ] Price prediction model
- [ ] User history & personalization
- [ ] React frontend dashboard
- [ ] Multi-language support

---

## 👨‍💻 Author

Built by **[Your Name]**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME)
