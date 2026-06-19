# Backend - News RAG (FastAPI)

The REST API service that wraps the RAG pipeline. This is where everything runs:
classify the query -> filter by metadata -> retrieve from ChromaDB -> aggregate
sentiment -> generate a **grounded** answer via the LLM. The frontend only calls
this API.

## Folder contents

| File | Description |
|------|-------------|
| `main.py` | FastAPI app: lifespan (loads models once at startup), `/health` and `/ask` endpoints, reads API key from ENV, CORS. |
| `rag_core.py` | Core RAG pipeline (no UI): load components, classify, retrieve+dedupe, aggregate sentiment, grounding prompt, generate. |
| `requirements.txt` | Backend dependencies (FastAPI + models). |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Status + whether components (vector DB, embedding, classifier) are ready. |
| `POST` | `/ask` | RAG Q&A. JSON body -> answer + sources + sentiment summary. |

`POST /ask` body:

```json
{
  "question": "Apa yang terjadi dengan IHSG?",
  "provider": "groq",
  "k": 5,
  "force_category": "Auto",
  "sentiment_filter": "Semua"
}
```

Only `question` is required; the rest are optional. Field details + interactive
docs are available at `/docs`.

## How it works (brief)

- **Models are loaded once** at startup (FastAPI `lifespan`), not per request - the
  vector DB, embedding model, and classifier are kept in memory.
- **Retrieval:** fetch `k*4` candidates -> dedupe per article -> take k unique
  articles. The category filter applies when classifier confidence >= 0.5 (or for a
  manual category); the sentiment filter is optional. Auto mode falls back to no
  filter when results are too few.
- **Grounding:** the answer comes only from the context; anything beyond that ->
  an honest "not found".

## Configuration (ENV)

| ENV | Description |
|-----|-------------|
| `GROQ_API_KEY` | Required for the groq provider (default). |
| `GEMINI_API_KEY` | Required for the gemini provider (optional). |
| `CHROMA_DIR` | Override the ChromaDB folder location (default `../chroma_db`). Used when Docker mounts a volume. |
| `CLASSIFIER_PATH` | Override the classifier model location (default `../models/indobert_classifier`). |

API keys are read from ENV, **not** from the request body.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# check: http://localhost:8000/health
# docs:  http://localhost:8000/docs
```

Make sure `../chroma_db/` and `../models/indobert_classifier/` are present (produced
by the notebook pipeline) before starting.

## Run with Docker

From the project root, via docker-compose:

```bash
docker compose up --build
# backend: http://localhost:8000
```

Default port **8000**. In compose, `chroma_db` is mounted read-write and `models`
read-only; paths point to the volumes via `CHROMA_DIR` / `CLASSIFIER_PATH`.

> CPU-only torch note: to keep the Docker image small, install torch from the CPU
> index at build time (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).
> On a local machine that already has GPU torch, no change is needed.
