"""
================================================================================
batch_run.py - Runner pertanyaan RAG secara batch (tanpa UI)
================================================================================
Jalankan banyak pertanyaan lewat pipeline RAG tanpa buka UI Streamlit.
Tiap interaksi otomatis tercatat ke rag_logbook.jsonl (via rag_logger).

INCREMENTAL: pertanyaan yang SUDAH ada di rag_logbook.jsonl otomatis di-skip
(tidak di-generate ulang, tidak buang token). Jadi cukup jalankan perintah yang
sama tiap hari; yang sudah jalan dilewati, yang baru diproses.

Pertanyaan (100) disusun GROUNDED ke isi korpus (articles_with_sentiment.csv),
seimbang:
  - per KATEGORI : 15 pertanyaan no-filter tiap kategori (finance/health/inet/news/sport)
  - per SENTIMEN : 5 pertanyaan tiap sentimen (positif/netral/negatif), tersebar antarkategori
  - OVERALL      : seluruh pertanyaan no-filter (75 grounded + 10 out-of-corpus)
  - 10 out-of-corpus untuk uji anti-halusinasi

PENTING soal kuota:
  Script ini MEN-GENERATE jawaban (1 call LLM per pertanyaan). Pakai Groq:
      set GROQ_API_KEY=key_groq_kamu
      python batch_run.py --provider groq
  Groq 70b ~100rb token/hari (~40-50 pertanyaan). Kena limit -> berhenti;
  besok jalankan perintah yang SAMA, sisanya lanjut otomatis (di-skip yang sudah).

Setelah selesai:
    python evaluate_logbook.py --judge groq
================================================================================
"""

import os
import json
import time
import argparse
from pathlib import Path

from rag_logger import log_interaction

LOGBOOK_PATH = Path("rag_logbook.jsonl")


def interaction_key(question, sentiment):
    """Identitas stabil sebuah interaksi.

    Harus identik dengan definisi di evaluate_logbook.py supaya logika skip
    (incremental) dan evaluasi merujuk ke interaksi yang sama.
    """
    return f"{(question or '').strip()}||{sentiment if sentiment else ''}"


def load_done_keys(path=LOGBOOK_PATH):
    """Kumpulan key interaksi yang SUDAH ada di logbook -> dipakai untuk skip."""
    done = set()
    if not Path(path).exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(interaction_key(d.get("question"), d.get("sentiment")))
    return done


# ============================================================
# KONFIGURASI
# ============================================================
# Path di-anchor ke root project (naik dari evaluation/), sama seperti yang
# dipakai backend, supaya menunjuk ke vector DB dan model yang sama.
PROJECT_ROOT    = Path(__file__).resolve().parents[1]  # naik dari evaluation/ ke root
CHROMA_DIR      = PROJECT_ROOT / "chroma_db"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "indobert_classifier"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "news_articles"
GEMINI_MODEL    = "gemini-2.5-flash"
GROQ_MODEL      = "llama-3.3-70b-versatile"
CONF_THRESHOLD  = 0.5

SYSTEM_PROMPT = (
    "Kamu asisten berita berbahasa Indonesia. Jawab pertanyaan HANYA berdasarkan konteks artikel "
    "yang diberikan di bawah. Jika informasinya tidak ada di konteks, katakan dengan jujur bahwa "
    "kamu tidak menemukannya dalam artikel yang tersedia - JANGAN mengarang. Jawab ringkas, jelas, "
    "dalam bahasa Indonesia, dan sebutkan nomor sumber yang kamu pakai (misal [1], [2])."
)

# ============================================================
# 100 PERTANYAAN (GROUNDED ke korpus)
# Format: (pertanyaan, sentiment_filter)  ; sentiment None = overall/no-filter
# ============================================================
QUESTIONS = [
    # ============ FINANCE — 11 no-filter ============
    ("Bagaimana pergerakan nilai tukar rupiah terhadap dolar AS?", None),
    ("Bagaimana kinerja IHSG belakangan ini?", None),
    ("Apa kebijakan ekspor terbaru yang dibahas pemerintah?", None),
    ("Apa kabar terbaru soal Menteri Keuangan Purbaya?", None),
    ("Apa isi Paket Stimulus Ekonomi Semester II 2026 dari pemerintah?", None),
    ("Bagaimana aksi jual bersih investor asing di pasar saham Indonesia?", None),
    ("Apa dugaan manipulasi harga ekspor CPO ke Amerika Serikat lewat Singapura?", None),
    ("Apa kebijakan ekonomi Presiden Prabowo yang terbaru?", None),
    ("Apa kabar soal ekspor nikel atau Nickel Pig Iron?", None),
    ("Apa dampak Indonesia didepak dari indeks FTSE Russell?", None),
    ("Bagaimana penyaluran gaji ketiga belas pensiunan ASN oleh TASPEN?", None),

    # ============ HEALTH — 11 no-filter ============
    ("Apa itu wabah hantavirus yang diberitakan?", None),
    ("Apa kaitan hantavirus dengan kapal pesiar Hondius?", None),
    ("Bagaimana perkembangan kasus Ebola di Afrika?", None),
    ("Apa kabar terbaru soal penyakit kanker di pemberitaan?", None),
    ("Apa kata Menteri Kesehatan soal obesitas dan umur panjang?", None),
    ("Apa kabar soal penyakit ginjal yang dibahas?", None),
    ("Apa peringatan BPOM terbaru soal obat atau makanan?", None),
    ("Bagaimana kondisi imunisasi anak di Aceh?", None),
    ("Apa kabar soal gangguan kesehatan mental menurut berita?", None),
    ("Apa risiko atau fakta soal kesehatan jantung yang diberitakan?", None),
    ("Apa isu kesehatan terkait program internship dokter?", None),

    # ============ INET — 11 no-filter ============
    ("Apa fitur keamanan iPhone untuk melindungi dari pencurian?", None),
    ("Apa perbedaan MacBook terbaru yang dirilis?", None),
    ("Apa fitur atau produk baru dari Samsung yang diberitakan?", None),
    ("Gadget OPPO apa yang baru rilis di Indonesia?", None),
    ("Apa konflik antara OpenAI dan Apple?", None),
    ("Apa kabar terbaru soal Meta di pemberitaan?", None),
    ("Game apa yang sedang menjadi sorotan bulan ini?", None),
    ("Apa kabar soal varian Ultra dari produk gadget?", None),
    ("Apa berita soal PHK karyawan perusahaan teknologi?", None),
    ("Apa produk Apple terbaru yang dibahas?", None),
    ("Apa smartphone baru yang rilis di Indonesia?", None),

    # ============ NEWS — 11 no-filter ============
    ("Bagaimana persiapan Idul Adha dan kurban tahun ini?", None),
    ("Bagaimana antusiasme warga membagikan dan menerima hewan kurban Idul Adha?", None),
    ("Apa kabar terbaru soal konflik Iran?", None),
    ("Apa hasil pertemuan Presiden Prabowo dengan Macron?", None),
    ("Apa kabar soal Donald Trump di pemberitaan?", None),
    ("Apa berita soal Israel dan Palestina bulan ini?", None),
    ("Apa kasus yang sedang ditangani polisi belakangan ini?", None),
    ("Apa kabar soal pemeriksaan hewan kurban?", None),
    ("Apa kejadian besar di Bogor yang diberitakan?", None),
    ("Apa berita viral soal kerbau 'Donald Trump'?", None),
    ("Apa kabar terbaru soal dunia pendidikan atau sekolah di Indonesia?", None),

    # ============ SPORT — 11 no-filter ============
    ("Bagaimana performa Marc Marquez di MotoGP 2026?", None),
    ("Apa kabar Francesco Bagnaia di MotoGP musim 2026?", None),
    ("Bagaimana performa Jorge Martin dan tim Aprilia di MotoGP?", None),
    ("Bagaimana penampilan Veda Ega di Moto3?", None),
    ("Apa hasil Indonesia di Uber Cup 2026?", None),
    ("Apa hasil Indonesia di Thomas Cup 2026?", None),
    ("Bagaimana hasil Thailand Open 2026 untuk Indonesia?", None),
    ("Bagaimana penampilan Jonatan Christie di turnamen bulu tangkis?", None),
    ("Apa kabar Alex Marquez di MotoGP?", None),
    ("Bagaimana performa Ginting belakangan ini?", None),
    ("Apa hasil ganda putra Leo/Daniel bulan ini?", None),

    # ============ SENTIMEN: POSITIF — 11 (tersebar antarkategori) ============
    ("Prestasi atau kemenangan apa yang diraih atlet bulu tangkis Indonesia?", "positif"),
    ("Kabar baik apa soal pembalap Indonesia di MotoGP atau Moto3?", "positif"),
    ("Ajang olahraga apa yang sukses digelar bulan ini?", "positif"),
    ("Game atau produk hiburan digital apa yang sedang populer di Indonesia?", "positif"),
    ("Gadget baru apa yang rilis di Indonesia dan mendapat respons baik?", "positif"),
    ("Kabar baik apa soal penyaluran dana atau bantuan pemerintah?", "positif"),
    ("Pencapaian positif apa di sektor ekonomi atau ekspor?", "positif"),
    ("Acara atau kegiatan masyarakat apa yang berlangsung meriah?", "positif"),
    ("Kabar baik apa soal pembangunan atau akses untuk warga?", "positif"),
    ("Kabar baik apa soal akses atau layanan kesehatan?", "positif"),
    ("Inisiatif kesehatan positif apa yang diberitakan?", "positif"),

    # ============ SENTIMEN: NEGATIF — 11 ============
    ("Wabah atau penyakit apa yang mengkhawatirkan bulan ini?", "negatif"),
    ("Peringatan kesehatan serius apa yang dikeluarkan otoritas?", "negatif"),
    ("Masalah kesehatan apa yang menimpa masyarakat?", "negatif"),
    ("Peristiwa tragis atau kecelakaan apa yang diberitakan?", "negatif"),
    ("Konflik atau ketegangan internasional apa yang memburuk?", "negatif"),
    ("Kasus kriminal apa yang meresahkan bulan ini?", "negatif"),
    ("Masalah atau konflik apa di dunia teknologi belakangan ini?", "negatif"),
    ("Berita buruk apa soal perusahaan teknologi atau karyawannya?", "negatif"),
    ("Kabar buruk apa soal ekonomi atau pasar saham?", "negatif"),
    ("Penurunan atau kerugian apa yang terjadi di sektor keuangan?", "negatif"),
    ("Insiden atau crash apa di dunia balap MotoGP?", "negatif"),

    # ============ SENTIMEN: NETRAL — 11 ============
    ("Pernyataan resmi apa soal kebijakan ekspor atau pasar saham?", "netral"),
    ("Data atau statistik ekonomi apa yang dirilis bulan ini?", "netral"),
    ("Pengumuman resmi apa dari otoritas keuangan?", "netral"),
    ("Pengumuman atau pernyataan resmi apa dari pemerintah bulan ini?", "netral"),
    ("Jadwal atau aturan resmi apa yang diumumkan terkait Idul Adha?", "netral"),
    ("Pernyataan pejabat apa soal isu nasional bulan ini?", "netral"),
    ("Apa tips atau cara memanfaatkan fitur perangkat teknologi terbaru?", "netral"),
    ("Informasi faktual apa soal fitur atau layanan digital baru?", "netral"),
    ("Bagaimana klasemen sementara pembalap di MotoGP musim 2026?", "netral"),
    ("Bagaimana hasil pertandingan bulu tangkis Indonesia di turnamen terbaru?", "netral"),
    ("Pernyataan resmi WHO atau Kemenkes soal kesehatan apa?", "netral"),

    # ============ OUT-OF-CORPUS — 12 (uji anti-halusinasi) ============
    ("Siapa pemenang Piala Dunia 2022?", None),
    ("Berapa harga saham Apple hari ini?", None),
    ("Bagaimana prediksi cuaca Jakarta besok?", None),
    ("Apa resep rendang yang enak?", None),
    ("Kapan film Avengers berikutnya tayang?", None),
    ("Siapa presiden Amerika Serikat saat ini?", None),
    ("Apa ibukota negara Australia?", None),
    ("Bagaimana cara membuat kue bolu?", None),
    ("Siapa peraih medali emas Olimpiade 2024?", None),
    ("Apa rumus luas lingkaran?", None),
    ("Siapa penemu bola lampu?", None),
    ("Berapa jumlah pemain dalam satu tim sepak bola?", None),
]


# ============================================================
# KOMPONEN (dimuat sekali)
# ============================================================
_collection = _embed = _clf = None


def load_components():
    """Muat embedding model + ChromaDB + classifier sekali ke variabel modul."""
    global _collection, _embed, _clf
    if _collection is not None:
        return
    import chromadb
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers import pipeline

    print("[init] embedding model...")
    _embed = SentenceTransformer(EMBEDDING_MODEL)
    print("[init] ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = client.get_collection(COLLECTION_NAME)
    print("[init] classifier...")
    device = 0 if torch.cuda.is_available() else -1
    _clf = pipeline("text-classification", model=str(CLASSIFIER_PATH),
                    tokenizer=str(CLASSIFIER_PATH), device=device,
                    truncation=True, max_length=256)


# ============================================================
# PIPELINE
# ============================================================
# Logika RAG yang sama dengan layanan utama, dijalankan langsung di script ini
# (tanpa lewat API) supaya batch runner mandiri dan mudah dijalankan dari CLI.

def generate(provider, api_key, system_prompt, user_prompt, temperature=0.3, max_retries=4):
    """Panggil LLM dengan retry backoff. Kuota harian habis -> error yang jelas."""
    for attempt in range(max_retries):
        try:
            if provider == "gemini":
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=api_key)
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt, temperature=temperature))
                return resp.text
            else:
                from groq import Groq
                client = Groq(api_key=api_key)
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}],
                    temperature=temperature)
                return resp.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg or "per day" in msg.lower():
                raise RuntimeError(
                    f"Kuota HARIAN {provider} habis. Pakai --provider groq "
                    "(atau tunggu reset).") from e
            if ("503" in msg or "UNAVAILABLE" in msg or "429" in msg) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def classify_query(query):
    """Prediksi kategori dari pertanyaan. Return (label, skor)."""
    r = _clf(query)[0]
    return r["label"], r["score"]


def retrieve(query, k=5, where=None):
    """Ambil top-k chunk paling relevan dari ChromaDB, dedupe per artikel."""
    q_emb = _embed.encode([query], normalize_embeddings=True).tolist()
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


def rag_answer(query, provider, api_key, k=5, sentiment_filter=None):
    """Alur RAG: classify -> filter (kategori + sentimen) -> retrieve -> generate -> log."""
    cat, cat_conf = classify_query(query)

    conds = []
    if cat_conf >= CONF_THRESHOLD:
        conds.append({"kategori": cat})
    if sentiment_filter:
        conds.append({"sentiment": sentiment_filter})
    where = conds[0] if len(conds) == 1 else ({"$and": conds} if conds else None)

    chunks = retrieve(query, k=k, where=where)
    if sentiment_filter is None and where is not None and len(chunks) < k:
        chunks = retrieve(query, k=k, where=None)

    backend = f"{provider}:{GEMINI_MODEL if provider == 'gemini' else GROQ_MODEL}"

    if not chunks:
        nf = "Maaf, tidak ada artikel relevan yang ditemukan untuk pertanyaan ini."
        log_interaction(question=query, answer=nf, contexts=[], category=cat,
                        sentiment=sentiment_filter, backend=backend)
        return nf, []

    context = build_context(chunks)
    user_prompt = f"Konteks artikel:\n\n{context}\n\nPertanyaan: {query}\n\nJawaban:"
    answer = generate(provider, api_key, SYSTEM_PROMPT, user_prompt)

    log_interaction(question=query, answer=answer, contexts=[c["text"] for c in chunks],
                    category=cat, sentiment=sentiment_filter, backend=backend)
    return answer, chunks


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch runner pertanyaan RAG (incremental via logbook)")
    ap.add_argument("--provider", choices=["gemini", "groq"], default="groq")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None,
                    help="maksimal berapa pertanyaan BARU dijalankan run ini (hemat kuota)")
    ap.add_argument("--force", action="store_true",
                    help="abaikan cache logbook, generate ulang semua (bisa bikin duplikat)")
    args = ap.parse_args()

    key_env = "GEMINI_API_KEY" if args.provider == "gemini" else "GROQ_API_KEY"
    if key_env not in os.environ:
        raise SystemExit(f"Set dulu {key_env}. Contoh (cmd): set {key_env}=...")
    api_key = os.environ[key_env]

    # Tentukan pertanyaan yang perlu dijalankan: skip yang sudah ada di logbook
    # (kecuali --force), lalu batasi dengan --limit kalau diminta.
    done = set() if args.force else load_done_keys()
    total = len(QUESTIONS)
    todo = [(q, s) for (q, s) in QUESTIONS if interaction_key(q, s) not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"\n=== Batch run (provider={args.provider}) ===")
    print(f"    total pertanyaan : {total}")
    print(f"    sudah di logbook : {total - len([1 for q, s in QUESTIONS if interaction_key(q, s) not in done])} (di-skip)")
    print(f"    akan dijalankan  : {len(todo)} pertanyaan baru\n")

    if not todo:
        print("Semua pertanyaan sudah ada di logbook. Tidak ada yang perlu dijalankan.")
        print("Lanjut evaluasi:  python evaluate_logbook.py --judge groq")
        raise SystemExit(0)

    load_components()

    ran = 0
    for q, sent in todo:
        tag = f"[sent:{sent}]" if sent else "[overall]"
        try:
            ans, chunks = rag_answer(q, args.provider, api_key, k=args.k, sentiment_filter=sent)
            preview = (ans or "").replace("\n", " ")[:58]
            ran += 1
            print(f"[{ran:03d}/{len(todo)}] {tag:<12} {q[:38]:<38} -> {len(chunks)} src | {preview}...")
        except Exception as e:
            print(f"[err] '{q[:30]}...': {e}")
            if "Kuota HARIAN" in str(e):
                print("\n>> Kuota habis. Berhenti. Besok jalankan perintah yang SAMA — "
                      "yang sudah jalan otomatis di-skip.")
                break
        time.sleep(1)

    print(f"\nSelesai. {ran} pertanyaan baru tercatat di rag_logbook.jsonl")
    print("Lanjut evaluasi:  python evaluate_logbook.py --judge groq")