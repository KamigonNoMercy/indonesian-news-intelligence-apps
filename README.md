# 📰 News RAG - Indonesian News Q&A with Classification + Sentiment

This repository contains an **end-to-end Retrieval-Augmented Generation (RAG)**
question-answering system over **Indonesian news articles**, combined with two NLP
components: **category classification** (fine-tuned IndoBERT) and **sentiment
analysis** (fine-tuned IndoBERT). Each query is classified first, then article
search is filtered by category + sentiment, before the answer is composed by an LLM
in a **grounded** way (only from retrieved context, anti-hallucination).

👉 **Live demo:** [News Intelligence Assistant on Hugging Face Spaces](https://huggingface.co/spaces/Kamigon/news-rag-apps)
👉 The full pipeline - **scraping, preprocessing, model fine-tuning, retrieval, and the FE/BE/Docker app** - is my own work.

> ⚠️ **Note on empty folders:** the `models/` and `chroma_db/` folders are kept
> empty in this GitHub repo on purpose - their generated contents exceed GitHub's
> 25 MB web-upload limit. They are regenerated automatically when the notebooks /
> pipeline are run (see [Repository Folder Notes](#-repository-folder-notes)).

---

## ✨ Features

- End-to-end pipeline: **scrape -> preprocess + sentiment -> fine-tune classifier -> chunk + build ChromaDB -> RAG serve (FastAPI + Streamlit)**
- **Three NLP components** working together in one query path:
  - **Category classifier** (fine-tuned IndoBERT) - routes the query to a category
  - **Sentiment model** (fine-tuned IndoBERT) - labels each article at index time
  - **RAG** (ChromaDB + embedding + LLM) - retrieves and answers, grounded
- **Metadata-aware retrieval:** vector search is filtered by predicted category and
  optional sentiment, then deduplicated per article.
- **Grounded generation:** the LLM may only answer from retrieved context;
  out-of-coverage questions get an honest "not found" (anti-hallucination).
- **Clean BE/FE split:** the same pipeline runs locally (docker-compose, 2 containers)
  and as a public single-container Hugging Face Space.

---

## 🎯 Goals

1. Answer Indonesian-language questions about current news, **grounded in real
   articles** (not the LLM's internal knowledge), with sources cited.
2. Use metadata (category & sentiment) to **narrow the search** for more relevant
   answers, and to **summarize the sentiment** of the sources used.
3. Ensure the system **honestly says "not found"** when a question is outside the
   data coverage, instead of fabricating an answer.

---

## 🏗️ Architecture

```
User Query
   |
   v
[Classify Query]  --(predicted category)--+
   |                                       |
   v                                       v
[Embed Query] -> [Vector Search] -> [Filter by Metadata: category + sentiment]
                                          |
                                          v
                                  [Aggregate Sentiment]
                                          |
                                          v
                                  [LLM Generate Answer]
                                          |
                                          v
                          Answer + Sources + Sentiment Summary
```

---

## 📊 Dataset

- **Source:** **Detik.com** (5 sub-channels: news, finance, sport, inet, health)
- **Time range:** **1 May - 29 May 2026**
- **Raw scraped:** 2000 articles -> **1840** after preprocessing
- **Split:** train 1472 / val 184 / test 184
- **Categories (5):** news 397, finance 375, sport 366, inet 359, health 343
- **Sentiment:** neutral 898, negative 494, positive 448

Each article stores: `article_id`, `judul` (title), `isi` (body), `kategori`
(category), `url`, `tanggal` (date), `n_words`, `sentiment`, `sentiment_score`.

---

## 🧠 Models (for metadata & retrieval)

| Role | Model | Notes |
|---|---|---|
| **Category classification** | IndoBERT (`indobenchmark/indobert-base-p1`) fine-tuned | 5 classes; accuracy **~0.957**, F1-macro ~0.956 -> `models/indobert_classifier` |
| **Sentiment analysis** | IndoBERT (`indobert-base-p1`) fine-tuned on IndoNLU **SmSA** | 3 classes; accuracy **~0.914**, F1-macro ~0.888 -> `models/sentiment_indobert` |
| **Embedding** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 dims, normalized (cosine) |
| **LLM generator** | Groq `llama-3.3-70b-versatile` (default) or Gemini `gemini-2.5-flash` | API key from ENV |

👉 Category is predicted **at query time**; sentiment is **precomputed per article**
during preprocessing and stored as metadata (not recomputed at inference).

---

## 🗄️ ChromaDB Configuration

| Item | Value |
|---|---|
| Collection | `news_articles` |
| Distance metric | cosine (`hnsw:space = cosine`) |
| Chunking | max **200 words/chunk**, **1-sentence** overlap |
| Total chunks | **4448** |
| Metadata per chunk | `article_id`, `chunk_id`, `chunk_index`, `judul`, `kategori`, `sentiment`, `sentiment_score`, `url`, `tanggal` |

## 🔧 RAG Configuration

| Item | Value |
|---|---|
| Top-k sources | 5 (configurable 3-10) |
| Retrieval | fetch `k*4` candidates -> dedupe per article -> take k unique |
| Category filter | applied when classifier confidence **>= 0.5**, or manual category |
| Sentiment filter | optional (positive/neutral/negative) |
| Fallback | Auto mode: if filtered results are too few, retry without filter |
| Grounding | system prompt forbids answering outside context; must cite source numbers |
| LLM retry | backoff (1,2,4 s) for 503/429 errors |

---

## 🖥️ Backend (FastAPI)

Wraps the RAG pipeline into a REST API. No Streamlit/UI here.

- **Models loaded once** at startup (FastAPI `lifespan`), not per request - vector
  DB, embedding model, and classifier are kept in memory.
- **Endpoints:** `GET /health` (status + readiness) and `POST /ask` (JSON body ->
  answer + sources + sentiment summary).
- **API key** read from ENV (`GROQ_API_KEY` / `GEMINI_API_KEY`), not from the body.
- **Paths** (chroma_db & models) overridable via ENV (`CHROMA_DIR`, `CLASSIFIER_PATH`).
- **CORS** enabled (tighten in production). Default port **8000**.

```bash
# from backend/
uvicorn main:app --reload --port 8000   # docs: http://localhost:8000/docs
```

## 🎨 Frontend (Streamlit)

Pure presentation - it does **not** load models, needs no GPU, and does not import
torch/chromadb. All RAG work is done by the backend; the frontend calls it over HTTP.

- Shows answer, source list (title, category, sentiment, distance), and a sentiment bar.
- Sidebar: backend health indicator, category/sentiment filters, number of sources `k`.
- Backend address from ENV `BACKEND_URL` (default `http://localhost:8000`). Custom
  dark theme. Default port **8501**.

```bash
# from frontend/ (backend must already be up)
streamlit run app.py
```

## 🐳 Docker

Two services orchestrated via `docker-compose.yml` at the project root.

| Service | Build | Port | Notes |
|---|---|:---:|---|
| `backend` | `./backend` | 8000 | FastAPI + models; heavy (torch, transformers) |
| `frontend` | `./frontend` | 8501 | Streamlit; lightweight, no models |

- **Volumes:** `chroma_db` mounted **read-write** (ChromaDB writes lock/WAL on
  connect); `models` mounted **read-only**.
- **ENV:** API keys passed through from `.env`; paths point to volume locations.
- **Networking:** the frontend points to the backend by service name
  (`BACKEND_URL=http://backend:8000`, Docker internal DNS - not localhost), with
  `depends_on: backend` ordering startup.

```bash
docker compose up --build     # run everything
docker compose down           # stop
```

---

## 🗃️ Repository Folder Notes

The `models/` and `chroma_db/` folders are intentionally kept **empty** in this
GitHub repository - only the folder structure is committed. Their generated contents
are too large for GitHub's web interface (some files are **larger than 25 MB**).

When the notebooks / pipeline are run locally, these folders are regenerated
automatically:

- `models/` -> the fine-tuned IndoBERT category and sentiment models.
- `chroma_db/` -> the generated ChromaDB vector database.

So a fresh clone may show these folders as empty at first; they get populated after
running the corresponding notebooks / pipeline steps.

---

## 🚀 Deployment to Hugging Face Spaces

Live: **https://huggingface.co/spaces/Kamigon/news-rag-apps**

This repo (local version) uses **docker-compose with 2 containers**. For HF Spaces,
a **single-container** variant is used. Differences:

| Aspect | Local (this repo) | HF Spaces |
|---|---|---|
| Orchestration | docker-compose, **2 containers** | **1 container**, 2 processes via `start.sh` |
| Public port | backend 8000 + frontend 8501 | only **7860** (Streamlit); FastAPI internal on `localhost:8000` |
| Requirements | per-folder (`backend/`, `frontend/`) | one **combined `requirements.txt`** at root |
| LLM provider | Groq **or** Gemini | **Groq only** (Gemini option removed from UI & API) |
| torch | per machine (can be GPU) | **CPU-only** (CPU index) for a smaller image |
| Model & ChromaDB | host volumes (RW/RO) | **baked** into the image, tracked with **git-lfs** |
| API key | from `.env` | **Secret** under Settings -> Variables and secrets |
| Misc | - | non-root user uid 1000 (HF requirement), `frontend/.streamlit/config.toml` |

👉 The core RAG pipeline, models, and ChromaDB configuration are **identical** to the
local version; only the packaging and deployment differ. On the Space, HF builds from
the `Dockerfile`, then `start.sh` starts FastAPI internally (`127.0.0.1:8000`), waits
for `/health`, and only then starts Streamlit on the public port **7860**.

> ⚠️ **Why Groq-only on HF:** Groq is currently the only provider evaluated end-to-end,
> so the public Space ships with Groq alone to keep the demo matching the measured
> behavior. Re-enabling Gemini is planned once it is evaluated under the same protocol
> (see [Future Work](#-future-work)).

---

## 🧪 Evaluation

Automatic, reference-free evaluation (**LLM-as-a-judge, similar to RAGAS**) over 100
interactions: faithfulness, answer relevancy, and anti-hallucination.

| Metric | Score |
|---|:---:|
| Classifier accuracy (category) | **0.9565** (F1 0.9560) |
| Sentiment model accuracy | **0.9140** (F1 0.8882) |
| Anti-hallucination (refusal correctness) | **1.000** |
| Overall faithfulness / relevancy | ~0.63 / ~0.65 |

👉 **Anti-hallucination is perfect (1.000)** - every "not found" refusal was correct;
the system did not fabricate answers for out-of-coverage questions. Full method and
per-category / per-sentiment breakdown are in `evaluasi_news_rag.md`.

---

## 🔮 Future Work

1. **Re-enable the Gemini provider** after evaluating it with the **RAGAS** pipeline
   under the same protocol as Groq. Once its faithfulness / relevancy /
   anti-hallucination are measured and comparable, add `GEMINI_API_KEY` to the HF
   Space and restore the provider switch in the UI.
2. **Improve weak segments** - the `inet` category and positive/neutral sentiment
   score lowest; expand/clean the corpus for those slices and revisit chunking.
3. **Expand data coverage** - the current knowledge base is a single month
   (1-29 May 2026); add a recurring scrape to keep the index fresh.
4. **Reranking** - add a cross-encoder reranking step after vector search to push the
   most relevant chunk to the top before generation.
5. **Production hardening** - tighten CORS to specific origins, add request logging /
   rate limiting, and cache repeated queries.

---

## 🧾 Conclusion

The system delivers a working Indonesian news Q&A assistant that is **grounded by
design**: it classifies the query, filters retrieval by category and sentiment, and
forces the LLM to answer only from retrieved context. The two supporting models are
strong (classifier accuracy ~0.957, sentiment accuracy ~0.914), and the headline RAG
result - **perfect anti-hallucination (1.000)** - shows the grounding strategy works:
the assistant refuses out-of-coverage questions instead of making things up.
Faithfulness and relevancy are solid but mid-range (~0.63 / ~0.65), pointing to
retrieval quality (not fabrication) as the next lever. The clean backend/frontend
split makes the same pipeline runnable both locally and as a public Hugging Face Space.

---

## 👥 Contributor (NLP Group C)

- Surya Dharma Putra
- Krisna Fery Rahmantya
- Agil Setiawan
- Khaerani Arista Dewi

---

## ⚙️ Stack

- **Scraping:** `requests` + `BeautifulSoup`
- **Classification & sentiment:** IndoBERT (`indobenchmark/indobert-base-p1`) fine-tuned
- **Embedding:** `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2)
- **Vector DB:** ChromaDB
- **LLM:** Groq (Llama 3.3 70B) or Google Gemini (2.5 Flash)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **Deploy:** Docker + docker-compose (local) / single-container Docker Space (HF)

---

## ⚠️ Repository Folder Notes

The `models/` and `chromadb/` folders are intentionally kept empty in this GitHub
repository. Their generated contents are too large to upload through the GitHub
web interface because some files are **larger than 25 MB**. Only the folder
structure is kept in the repository.

When the notebooks / pipeline are run locally, these folders will be generated
again automatically:

- `models/` will contain the fine-tuned IndoBERT category and sentiment models.
- `chromadb/` will contain the generated ChromaDB vector database.

Therefore, a fresh clone may show these folders as empty at first, but they will
be populated after running the corresponding notebooks / pipeline steps.
