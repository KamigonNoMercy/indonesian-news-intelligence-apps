"""
================================================================================
rag_core.py - Inti pipeline RAG 
================================================================================
Modul ini berisi seluruh logika RAG:
muat komponen -> klasifikasi -> retrieve+dedupe -> agregasi sentimen ->
grounding prompt -> generate.

Dipakai oleh:
  - backend/main.py (FastAPI) -> membungkus modul ini jadi REST API.
  - frontend/app.py           -> tidak mengimpor modul ini langsung; cukup
                                  memanggil API lewat HTTP.

Catatan desain:
  - Komponen (vector DB, embedding, classifier) dimuat SEKALI lewat
    init_components(), lalu disimpan di variabel modul. Tidak ada cache UI di sini.
  - Path di-anchor ke ROOT project (parent dari folder backend/), jadi
    chroma_db/ dan models/ tetap ketemu walau dijalankan dari direktori mana pun.
    Bisa juga di-override lewat env var (CHROMA_DIR / CLASSIFIER_PATH) saat Docker.
  - Tidak ada logging interaksi di sini; itu tanggung jawab layer API agar core
    tetap bersih dan mudah diuji.

Catatan path:
  rag_core.py ada di  <root>/backend/rag_core.py
  -> ROOT = parents[1]  (naik 1 level dari backend/)
================================================================================
"""

import os
import time
from pathlib import Path
from collections import Counter

# ============================================================
# KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parents[1]          # <root project>

# Path bisa di-override lewat env var (dipakai saat Docker pakai volume).
# Default: folder asli relatif ke root project (buat dev lokal).
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", ROOT / "chroma_db"))
CLASSIFIER_PATH = Path(os.environ.get("CLASSIFIER_PATH", ROOT / "models" / "indobert_classifier"))

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "news_articles"

GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"

CATEGORIES = ["finance", "health", "inet", "news", "sport"]
SENTIMENTS = ["positif", "netral", "negatif"]

SYSTEM_PROMPT = (
    "Kamu asisten berita berbahasa Indonesia. Jawab pertanyaan HANYA berdasarkan konteks artikel "
    "yang diberikan di bawah. Jika informasinya tidak ada di konteks, katakan dengan jujur bahwa "
    "kamu tidak menemukannya dalam artikel yang tersedia - JANGAN mengarang. Jawab ringkas, jelas, "
    "dalam bahasa Indonesia, dan sebutkan nomor sumber yang kamu pakai (misal [1], [2])."
)

# ============================================================
# KOMPONEN GLOBAL (dimuat sekali di startup)
# ============================================================

_collection = None      # koleksi ChromaDB
_embed_model = None      # SentenceTransformer
_classifier = None       # pipeline klasifikasi


def init_components():
    """Muat vector DB + embedding model + classifier SEKALI.

    Dipanggil saat startup (mis. di FastAPI lifespan). Aman dipanggil ulang;
    komponen yang sudah dimuat tidak dimuat lagi.
    """
    global _collection, _embed_model, _classifier

    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)

    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)

    if _classifier is None:
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        _classifier = pipeline(
            "text-classification",
            model=str(CLASSIFIER_PATH), tokenizer=str(CLASSIFIER_PATH),
            device=device, truncation=True, max_length=256,
        )

    return _collection, _embed_model, _classifier


def components_ready():
    """True kalau ketiga komponen sudah dimuat (dipakai oleh endpoint /health)."""
    return all(c is not None for c in (_collection, _embed_model, _classifier))


# ============================================================
# KLIEN LLM
# ============================================================

def _get_gemini_client(api_key):
    from google import genai
    return genai.Client(api_key=api_key)


def _get_groq_client(api_key):
    from groq import Groq
    return Groq(api_key=api_key)


def generate(provider, api_key, system_prompt, user_prompt, temperature=0.3, max_retries=4):
    """Panggil LLM sesuai provider, dengan retry backoff (1,2,4 dtk) untuk error 503/429."""
    for attempt in range(max_retries):
        try:
            if provider == "gemini":
                from google.genai import types
                client = _get_gemini_client(api_key)
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt, temperature=temperature))
                return resp.text
            else:  # groq (default)
                client = _get_groq_client(api_key)
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}],
                    temperature=temperature)
                return resp.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if ("503" in msg or "UNAVAILABLE" in msg or "429" in msg) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


# ============================================================
# TAHAP-TAHAP PIPELINE
# ============================================================

def classify_query(query):
    """Prediksi kategori dari pertanyaan. Return (label, skor)."""
    r = _classifier(query)[0]
    return r["label"], r["score"]


def retrieve(query, k=5, where=None):
    """Ambil top-k chunk paling relevan dari ChromaDB, dedupe per artikel.

    Mengambil k*4 kandidat lalu menyaring agar tiap artikel hanya muncul sekali
    (chunk dengan jarak terkecil yang menang), sampai terkumpul k artikel unik.
    """
    q_emb = _embed_model.encode([query], normalize_embeddings=True).tolist()
    res = _collection.query(query_embeddings=q_emb, n_results=k * 4, where=where)
    seen, out = set(), []
    for i in range(len(res["ids"][0])):
        m = res["metadatas"][0][i]
        if m["article_id"] in seen:
            continue
        seen.add(m["article_id"])
        out.append({"text": res["documents"][0][i], "distance": res["distances"][0][i], **m})
        if len(out) >= k:
            break
    return out


def build_context(chunks):
    """Susun chunk jadi teks konteks bernomor untuk prompt."""
    return "\n\n".join(
        f"[Sumber {i}] ({c['kategori']}/{c['sentiment']}) {c['judul']}\n{c['text']}"
        for i, c in enumerate(chunks, 1))


def rag_answer(query, provider="groq", api_key=None, k=5,
               force_category="Auto", sentiment_filter="Semua", conf_threshold=0.5):
    """Jalankan alur RAG lengkap untuk satu pertanyaan.

    Tahapan: klasifikasi kategori -> bangun filter metadata -> retrieve+dedupe
    (dengan fallback tanpa filter bila hasil sedikit di mode Auto) -> agregasi
    sentimen sumber -> susun prompt grounded -> generate jawaban.

    Return dict: answer, sources, kategori_query, kategori_conf, filter, sentimen.
    """
    if not components_ready():
        init_components()

    # 1. Klasifikasi kategori pertanyaan.
    cat, cat_conf = classify_query(query)

    # 2. Bangun filter metadata.
    #    Kategori manual (force_category) menang atas hasil classifier; classifier
    #    hanya dipakai bila keyakinannya >= conf_threshold. Sentimen difilter bila diminta.
    conds, used = [], {}
    if force_category != "Auto":
        conds.append({"kategori": force_category}); used["kategori"] = f"{force_category} (manual)"
    elif cat_conf >= conf_threshold:
        conds.append({"kategori": cat}); used["kategori"] = cat
    if sentiment_filter != "Semua":
        conds.append({"sentiment": sentiment_filter}); used["sentiment"] = sentiment_filter
    where = conds[0] if len(conds) == 1 else ({"$and": conds} if conds else None)

    # 3. Retrieve + dedupe. Di mode Auto, bila filter menghasilkan terlalu sedikit
    #    artikel, ulangi tanpa filter agar tetap ada konteks yang cukup.
    chunks = retrieve(query, k=k, where=where)
    if force_category == "Auto" and where is not None and len(chunks) < k:
        chunks = retrieve(query, k=k, where=None)
        used["fallback"] = "tanpa filter (hasil sedikit)"

    # Tidak ada artikel relevan -> jawab jujur "tidak ditemukan" (anti-halusinasi).
    if not chunks:
        return {
            "answer": "Maaf, tidak ada artikel relevan yang ditemukan untuk pertanyaan ini.",
            "sources": [], "kategori_query": cat, "kategori_conf": round(cat_conf, 2),
            "filter": used, "sentimen": {},
        }

    # 4. Agregasi sentimen dari sumber yang terambil.
    sent_summary = dict(Counter(c["sentiment"] for c in chunks))

    # 5-6. Susun prompt grounded lalu generate jawaban.
    context = build_context(chunks)
    user_prompt = f"Konteks artikel:\n\n{context}\n\nPertanyaan: {query}\n\nJawaban:"
    answer = generate(provider, api_key, SYSTEM_PROMPT, user_prompt)

    return {
        "answer": answer,
        "sources": [{"judul": c["judul"], "kategori": c["kategori"], "sentiment": c["sentiment"],
                     "url": c["url"], "distance": round(c["distance"], 3)} for c in chunks],
        "kategori_query": cat, "kategori_conf": round(cat_conf, 2),
        "filter": used, "sentimen": sent_summary,
    }