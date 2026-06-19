"""
================================================================================
rag_logger.py - Pencatat interaksi RAG ke logbook (JSONL)
================================================================================
Import modul ini di skrip yang men-generate jawaban (mis. batch_run.py, atau
layer API), lalu panggil `log_interaction(...)` setiap kali sistem selesai
menjawab. Tiap interaksi disimpan sebagai 1 baris JSON di rag_logbook.jsonl
(format JSONL: 1 objek per baris).

File logbook ini nanti dibaca oleh evaluate_logbook.py untuk dinilai otomatis.

CARA PASANG (tambahkan 2 hal saja):
    # 1) di bagian atas file:
    from rag_logger import log_interaction
    # 2) setelah jawaban final dihasilkan, sebelum/sesudah ditampilkan ke user:
    log_interaction(
        question=user_query,          # str: pertanyaan user
        answer=final_answer,          # str: jawaban akhir LLM
        contexts=retrieved_chunks,    # list[str] ATAU list[dict] berisi teks chunk
        category=predicted_category,  # str/None: hasil classifier (opsional)
        sentiment=sentiment_filter,   # str/None: filter sentimen jika ada (opsional)
        backend="gemini",             # str: model yang dipakai generate (opsional)
    )

`contexts` boleh:
  - list of str   -> ["teks chunk 1", "teks chunk 2", ...]
  - list of dict  -> [{"text": "...", "article_id": "..."}, ...]
Logger akan otomatis mengekstrak teksnya.
================================================================================
"""

import json
import threading
from datetime import datetime
from pathlib import Path

LOGBOOK_PATH = Path("rag_logbook.jsonl")
_lock = threading.Lock()  # aman jika Streamlit memanggil dari beberapa rerun


def _extract_text(contexts):
    """Normalisasi contexts jadi list[str] apa pun bentuk inputnya."""
    out = []
    for c in (contexts or []):
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, dict):
            out.append(c.get("text") or c.get("document") or c.get("content") or "")
        else:
            out.append(str(c))
    return [t for t in out if t]


def log_interaction(question, answer, contexts,
                    category=None, sentiment=None, backend=None,
                    extra=None):
    """
    Catat satu interaksi tanya-jawab ke logbook.

    Args:
        question : pertanyaan user (str)
        answer   : jawaban akhir sistem (str)
        contexts : chunk yang diambil retriever (list[str] atau list[dict])
        category : kategori hasil classifier (opsional)
        sentiment: filter sentimen yang dipakai (opsional)
        backend  : nama model generator, mis. "gemini-2.5-flash" (opsional)
        extra    : dict tambahan apa pun yang mau ikut dicatat (opsional)
    """
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "contexts": _extract_text(contexts),
        "category": category,
        "sentiment": sentiment,
        "backend": backend,
    }
    if extra:
        record["extra"] = extra
    line = json.dumps(record, ensure_ascii=False)
    # Lock supaya penulisan tetap utuh kalau dipanggil dari beberapa rerun/thread.
    with _lock:
        with open(LOGBOOK_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def count_logged():
    """Berapa interaksi yang sudah tercatat di logbook."""
    if not LOGBOOK_PATH.exists():
        return 0
    with open(LOGBOOK_PATH, encoding="utf-8") as f:
        return sum(1 for _ in f)