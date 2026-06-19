# Frontend - News RAG (Streamlit)

The user interface for the News RAG system. **Presentation only**: it does not
load models, needs no GPU, and does not import torch/chromadb. All RAG work is done
by the backend; the frontend simply calls it over HTTP.

```
[Streamlit UI] --POST /ask--> [FastAPI backend] --> [RAG: classify/retrieve/generate]
               <--- JSON ----
```

## Folder contents

| File | Description |
|------|-------------|
| `app.py` | Streamlit app: question input, calls the backend, displays answer + sources + sentiment. Custom dark theme. |
| `requirements.txt` | Lightweight dependencies (streamlit + requests only). |
| `Dockerfile` | Frontend image, run via docker-compose. |

## Features

- News Q&A: send a question -> show the LLM answer, source list (title, category,
  sentiment, distance), and a sentiment summary of the sources.
- Backend status indicator in the sidebar (checks `GET /health`).
- Sidebar filters: category (Auto / finance / health / inet / news / sport),
  sentiment (All / positive / neutral / negative), and number of sources `k` (3-10).
- LLM provider selection (groq / gemini) - the API key stays on the backend side.
- Clear "not found" handling for questions outside the data coverage.

## Configuration

| ENV | Default | Description |
|-----|---------|-------------|
| `BACKEND_URL` | `http://localhost:8000` | Address of the backend to call. In Docker set to `http://backend:8000`. |

API keys (`GROQ_API_KEY` / `GEMINI_API_KEY`) are **not** set here - that is the
backend's responsibility.

## Run locally

The backend must be up first (see `../backend/`).

```bash
pip install -r requirements.txt
streamlit run app.py
# open http://localhost:8501
```

## Run with Docker

From the project root, via docker-compose (builds backend + frontend together):

```bash
docker compose up --build
# frontend: http://localhost:8501
```

Default port **8501**. Within compose, the frontend waits for the backend
(`depends_on`) and points to it by service name (`BACKEND_URL=http://backend:8000`).
