"""
================================================================================
main.py - Layanan API (FastAPI) untuk News RAG
================================================================================
Membungkus rag_core jadi REST API. Model dimuat SEKALI saat startup (lifespan),
bukan tiap request.

Endpoint:
  GET  /health  -> cek status + apakah komponen sudah siap
  POST /ask     -> tanya-jawab RAG

Body POST /ask (JSON):
  {
    "question": "Apa yang terjadi dengan IHSG?",
    "provider": "groq",
    "k": 5,
    "force_category": "Auto",
    "sentiment_filter": "Semua"
  }
Detail tiap field (wajib/opsional, nilai default, dan pilihan yang valid) ada di
skema AskRequest di bawah, dan otomatis tampil di /docs.

API key diambil dari ENV (GROQ_API_KEY / GEMINI_API_KEY), bukan dari body.

JALANIN (dari folder backend/):
    uvicorn main:app --reload --port 8000
Lalu buka dokumentasi interaktif: http://localhost:8000/docs
================================================================================
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag_core


# ============================================================
# LIFESPAN: muat model sekali saat startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] memuat komponen RAG (vector DB, embedding, classifier)...")
    rag_core.init_components()
    print("[startup] komponen siap.")
    yield
    print("[shutdown] selesai.")


app = FastAPI(title="News Intelligence Assistant API", version="1.0.0", lifespan=lifespan)

# CORS: izinkan frontend memanggil API ini. Sempitkan origins-nya di produksi.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # ganti ke origin frontend spesifik saat produksi
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SKEMA REQUEST
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pertanyaan pengguna (wajib)")
    provider: str = Field("groq", description="groq (default) atau gemini")
    k: int = Field(5, ge=1, le=10, description="Jumlah sumber yang diambil")
    force_category: str = Field("Auto", description="Auto / finance / health / inet / news / sport")
    sentiment_filter: str = Field("Semua", description="Semua / positif / netral / negatif")


# ============================================================
# ENDPOINT
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "components_ready": rag_core.components_ready()}


@app.post("/ask")
def ask(req: AskRequest):
    # Ambil API key dari ENV sesuai provider.
    key_env = "GEMINI_API_KEY" if req.provider == "gemini" else "GROQ_API_KEY"
    api_key = os.environ.get(key_env)
    if not api_key:
        raise HTTPException(status_code=400,
                            detail=f"{key_env} belum di-set di environment.")
    try:
        result = rag_core.rag_answer(
            query=req.question, provider=req.provider, api_key=api_key, k=req.k,
            force_category=req.force_category, sentiment_filter=req.sentiment_filter,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses: {e}")