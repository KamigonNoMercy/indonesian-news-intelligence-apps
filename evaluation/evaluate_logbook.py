"""
================================================================================
evaluate_logbook.py - Evaluator RAG: incremental & tanpa referensi (reference-free)
================================================================================
Baca rag_logbook.jsonl (dihasilkan rag_logger.py saat pakai app.py / batch_run.py)
lalu skor tiap interaksi SECARA OTOMATIS dan TANPA jawaban acuan (reference-free):
 
  - Faithfulness      : apakah klaim di jawaban benar-benar didukung konteks?
  - Answer Relevancy  : apakah jawaban benar menjawab pertanyaannya?
  - Anti-halusinasi    : untuk jawaban "tidak ditemukan", apakah penolakannya tepat
                         (konteks memang tidak relevan) atau menyerah padahal bisa?
 
Tidak butuh gold set / ground-truth.
 
INCREMENTAL (fitur utama):
  Tiap interaksi yang SUDAH ada di logbook_eval.csv otomatis di-SKIP -> tidak ada
  call API, tidak buang token. Hanya interaksi BARU yang dikirim ke judge, lalu
  ditambahkan ke logbook_eval.csv. Jadi script ini bisa dijalankan tiap hari
  seiring logbook bertambah (lewat batch_run.py / app.py) tanpa pernah menskor
  ulang item yang sudah dinilai.
 
FILE OUTPUT:
  logbook_eval.csv         -> detail per-interaksi (terus bertambah tiap run)
  evaluation_summary.csv   -> ringkasan agregat (overall / per kategori / per sentimen)
 
CATATAN KUOTA:
  - Faithfulness + relevancy ditanya dalam SATU call -> request jadi separuh.
  - Budget token Groq dihitung per-model; judge pakai llama-3.1-8b-instant
    (budget terpisah dan lebih besar dari 70b yang dipakai batch_run untuk generate).
 
CARA PAKAI:
    set GROQ_API_KEY=key_groq_kamu
    python evaluate_logbook.py --judge groq
 
    set GEMINI_API_KEY=key_gemini_kamu
    python evaluate_logbook.py --judge gemini
 
    python evaluate_logbook.py --judge groq --limit 10
    python evaluate_logbook.py --ragas
    python evaluate_logbook.py --force        # skor ulang semua (abaikan cache)
================================================================================
"""
 
import os
import re
import csv
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
 
GEMINI_MODEL = "gemini-2.5-flash"
# Model judge Groq: 8b-instant -> budget token harian terpisah dari 70b yang
# dipakai batch_run untuk generate, dan jauh lebih longgar.
GROQ_MODEL   = "llama-3.1-8b-instant"
 
DETAIL_CSV  = "logbook_eval.csv"        # detail per-interaksi (terus bertambah)
SUMMARY_CSV = "evaluation_summary.csv"  # ringkasan agregat
 
# Batas karakter konteks di dalam prompt judge (hemat token harian).
MAX_CTX_CHARS = 2500
 
# Penanda (huruf kecil) yang menandai jawaban sebagai penolakan "tidak ditemukan".
NOT_FOUND_PATTERNS = ("tidak ditemukan", "tidak menemukan", "tidak tersedia",
                      "di luar cakupan", "tidak ada informasi",
                      "tidak ada artikel relevan", "maaf")
 
 
# ----------------------------------------------------------------------
# Logbook + kunci identitas
# ----------------------------------------------------------------------
def load_logbook(path, limit=None):
    """Baca logbook JSONL jadi list record (ambil `limit` terakhir bila diisi)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit:
        records = records[-limit:]
    return records
 
 
def interaction_key(question, sentiment):
    """Identitas stabil sebuah interaksi. Dipakai untuk skip item yang sudah dinilai.
 
    Harus identik dengan definisi di batch_run.py supaya logika skip sinkron.
    """
    return f"{(question or '').strip()}||{sentiment if sentiment else ''}"
 
 
def is_not_found(answer):
    """True bila jawaban terbaca sebagai penolakan "tidak ditemukan" (lihat NOT_FOUND_PATTERNS)."""
    a = (answer or "").lower()
    return any(p in a for p in NOT_FOUND_PATTERNS)
 
 
# ----------------------------------------------------------------------
# Muat baris yang sudah dinilai dari DETAIL_CSV (cache)
# ----------------------------------------------------------------------
def _to_float(v):
    """Ubah sel CSV jadi float, atau None untuk nilai kosong/invalid."""
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None
 
 
def load_existing(path):
    """Return dict: interaction_key -> baris yang sudah dinilai (untuk di-skip)."""
    cache = {}
    if not Path(path).exists():
        return cache
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("key") or interaction_key(row.get("question"), row.get("sentiment"))
            cache[key] = {
                "key": key,
                "timestamp": row.get("timestamp", ""),
                "question": row.get("question", ""),
                "category": row.get("category") or None,
                "sentiment": row.get("sentiment") or None,
                "not_found": int(row.get("not_found") or 0),
                "faithfulness": _to_float(row.get("faithfulness")),
                "answer_relevancy": _to_float(row.get("answer_relevancy")),
                "antihalusinasi_ok": _to_float(row.get("antihalusinasi_ok")),
            }
    return cache
 
 
# ----------------------------------------------------------------------
# Backend judge LLM + retry
# ----------------------------------------------------------------------
# Client dibuat lazy dan di-cache di level modul, jadi cuma init sekali.
_gemini_client = None
_groq_client = None
 
 
def _ask_gemini(prompt):
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text
 
 
def _ask_groq(prompt):
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = _groq_client.chat.completions.create(
        model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    return resp.choices[0].message.content
 
 
def ask(prompt, backend, max_retries=5):
    """Kirim prompt ke backend judge terpilih, dengan retry/backoff.
 
    Error kuota harian langsung dilempar (percuma di-retry); error sementara
    di-backoff sampai maksimal 30 detik.
    """
    fn = _ask_gemini if backend == "gemini" else _ask_groq
    for attempt in range(max_retries):
        try:
            return fn(prompt)
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg or "per day" in msg.lower():
                raise RuntimeError(
                    "Daily judge quota exhausted. Switch judge (--judge groq) "
                    "or wait for reset.") from e
            if attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt, 30)
            print(f"    [retry {attempt+1}] {msg[:80]} -> sleep {wait}s")
            time.sleep(wait)
 
 
# ----------------------------------------------------------------------
# Parsing skor
# ----------------------------------------------------------------------
def _norm(v):
    """Paksa skor judge ke rentang 0-1 (menangani skala 0-5 dan 0-100)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v > 1:
        v = v / 100 if v > 5 else v / 5
    return max(0.0, min(1.0, v))
 
 
def _parse_two_scores(text):
    """Ambil (faithfulness, relevancy) dari balasan judge.
 
    Diutamakan dari objek JSON; kalau gagal, fallback ke satu/dua angka pertama.
    """
    if not text:
        return None, None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            return _norm(d.get("faithfulness")), _norm(d.get("relevancy"))
        except json.JSONDecodeError:
            pass
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if len(nums) >= 2:
        return _norm(nums[0]), _norm(nums[1])
    if len(nums) == 1:
        return _norm(nums[0]), None
    return None, None
 
 
# ----------------------------------------------------------------------
# Metrik (digabung dalam satu call)
# ----------------------------------------------------------------------
def judge_answer(question, answer, contexts, backend):
    """Minta judge menilai faithfulness + relevancy dalam satu call. Return (f, r)."""
    context = "\n\n".join(contexts)[:MAX_CTX_CHARS]
    prompt = (
        "Anda evaluator sistem RAG. Nilai JAWABAN dengan dua kriteria:\n"
        "1. faithfulness: apakah SEMUA klaim dalam jawaban didukung KONTEKS "
        "(1=semua grounded, 0=mengarang).\n"
        "2. relevancy: apakah jawaban menjawab PERTANYAAN secara langsung "
        "(1=sangat relevan, 0=melenceng).\n"
        "Balas HANYA JSON: {\"faithfulness\": <0-1>, \"relevancy\": <0-1>}\n\n"
        f"PERTANYAAN:\n{question}\n\nKONTEKS:\n{context}\n\nJAWABAN:\n{answer}\n\nJSON:"
    )
    return _parse_two_scores(ask(prompt, backend))
 
 
def judge_context_relevant(question, contexts, backend):
    """Tanya judge apakah konteks memang menjawab pertanyaannya. Return 0-1."""
    context = ("\n\n".join(contexts))[:MAX_CTX_CHARS] if contexts else "(tidak ada konteks)"
    prompt = (
        "Apakah KONTEKS memuat informasi yang cukup untuk menjawab PERTANYAAN? "
        "Balas HANYA '1' (ya) atau '0' (tidak).\n\n"
        f"PERTANYAAN:\n{question}\n\nKONTEKS:\n{context}\n\nJawaban:"
    )
    return _norm((ask(prompt, backend) or "").strip()[:3])
 
 
def score_record(rec, backend):
    """Skor SATU record logbook lewat judge (melakukan call API).
 
    Penolakan dicek anti-halusinasinya (konteks seharusnya tidak relevan);
    item yang dijawab diskor faithfulness + relevancy.
    """
    q = rec.get("question", "")
    a = rec.get("answer", "")
    ctx = rec.get("contexts", [])
    nf = is_not_found(a)
    row = {
        "key": interaction_key(q, rec.get("sentiment")),
        "timestamp": rec.get("timestamp", ""),
        "question": q,
        "category": rec.get("category"),
        "sentiment": rec.get("sentiment"),
        "not_found": int(nf),
        "faithfulness": None,
        "answer_relevancy": None,
        "antihalusinasi_ok": None,
    }
    if nf:
        # Penolakan "benar" kalau konteks yang terambil memang tidak bisa menjawab.
        ctx_rel = judge_context_relevant(q, ctx, backend)
        row["antihalusinasi_ok"] = 1.0 if (ctx_rel is not None and ctx_rel < 0.5) else 0.0
    elif ctx:
        f, r = judge_answer(q, a, ctx, backend)
        row["faithfulness"] = f
        row["answer_relevancy"] = r
    return row
 
 
# ----------------------------------------------------------------------
# Run utama (incremental)
# ----------------------------------------------------------------------
def run_judge(records, backend, force=False):
    """Skor record baru (skip yang sudah di-cache), lalu tulis CSV detail + ringkasan."""
    cache = {} if force else load_existing(DETAIL_CSV)
    print(f"\n=== EVALUATE LOGBOOK (judge={backend}) — {len(records)} logbook records ===")
    print(f"    already scored in cache: {len(cache)}\n")
 
    new_count = skip_count = 0
    for i, rec in enumerate(records, 1):
        key = interaction_key(rec.get("question", ""), rec.get("sentiment"))
        if key in cache:
            skip_count += 1
            print(f"  [{i}/{len(records)}] SKIP (cached): {rec.get('question','')[:55]}")
            continue
        row = score_record(rec, backend)
        cache[key] = row
        new_count += 1
        print(f"  [{i}/{len(records)}] NEW  faith={row['faithfulness']} "
              f"rel={row['answer_relevancy']} nf={row['not_found']} "
              f"anti={row['antihalusinasi_ok']}")
        time.sleep(0.5)
 
    all_rows = list(cache.values())
    print(f"\n    scored this run: {new_count} new, {skip_count} skipped. "
          f"Total in results: {len(all_rows)}")
 
    summary = _report(all_rows)
    _save_detail(all_rows)
    _save_summary(summary)
    return all_rows
 
 
# ----------------------------------------------------------------------
# RAGAS (opsional)
# ----------------------------------------------------------------------
def run_ragas(records, backend):
    """Skor alternatif lewat library RAGAS (LLM + embedding Gemini)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
 
    key = os.environ["GEMINI_API_KEY"]
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=key))
    emb = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=key))
    data = {"question": [], "answer": [], "contexts": []}
    for rec in records:
        if is_not_found(rec.get("answer", "")) or not rec.get("contexts"):
            continue
        data["question"].append(rec["question"])
        data["answer"].append(rec["answer"])
        data["contexts"].append(rec["contexts"])
    if not data["question"]:
        print("RAGAS: no valid interactions.")
        return
    result = evaluate(Dataset.from_dict(data),
                      metrics=[faithfulness, answer_relevancy], llm=llm, embeddings=emb)
    print("\n[RAGAS]\n", result)
 
 
# ----------------------------------------------------------------------
# Metrik + laporan
# ----------------------------------------------------------------------
def _avg(vals):
    """Rata-rata nilai yang bukan None, atau None bila tidak ada sama sekali."""
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None
 
 
def _fmt(x):
    """Format skor ke 3 desimal, atau placeholder dash bila kosong."""
    return f"{x:.3f}" if x is not None else "  -  "
 
 
def _metrics(rows):
    """Agregasi sekelompok baris: jumlah + rata-rata faithfulness/relevancy/anti."""
    answered = [r for r in rows if not r["not_found"]]
    refused  = [r for r in rows if r["not_found"]]
    return {
        "n": len(rows),
        "answered": len(answered),
        "refused": len(refused),
        "faithfulness": _avg([r["faithfulness"] for r in answered]),
        "relevancy": _avg([r["answer_relevancy"] for r in answered]),
        "anti": _avg([r["antihalusinasi_ok"] for r in refused]),
    }
 
 
def _print_row(label, m):
    """Cetak satu baris metrik agregat ke konsol."""
    print(f"  {label:<22} | n={m['n']:<3} ans={m['answered']:<3} ref={m['refused']:<3} "
          f"| faith={_fmt(m['faithfulness'])} rel={_fmt(m['relevancy'])} anti={_fmt(m['anti'])}")
 
 
def _report(rows):
    """Cetak laporan ke konsol dan kembalikan baris ringkasan untuk ekspor CSV."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
 
    baseline = [r for r in rows if not r.get("sentiment")]
    filtered = [r for r in rows if r.get("sentiment")]
    summary = []
    gen_at = datetime.now().isoformat(timespec="seconds")
 
    def add(scope, group, m):
        summary.append({
            "scope": scope, "group": group, "n": m["n"],
            "answered": m["answered"], "refused": m["refused"],
            "faithfulness": _fmt(m["faithfulness"]).strip(),
            "answer_relevancy": _fmt(m["relevancy"]).strip(),
            "anti_hallucination": _fmt(m["anti"]).strip(),
            "generated_at": gen_at,
        })
 
    print("\n  [OVERALL]")
    mb = _metrics(baseline)
    _print_row("baseline (no filter)", mb); add("OVERALL", "baseline_no_filter", mb)
    if filtered:
        ma = _metrics(rows)
        _print_row("all (+filtered)", ma); add("OVERALL", "all_with_filtered", ma)
 
    # Helper: pecah baris berdasarkan kunci metadata (category / sentiment) lalu laporkan tiap grup.
    def group_block(key, scope_name, empty_label="no-filter"):
        groups = {}
        for r in rows:
            g = r.get(key) or empty_label
            groups.setdefault(g, []).append(r)
        if len(groups) == 1 and empty_label in groups:
            print(f"\n  --- By {scope_name} --- (no labelled data yet; all {empty_label})")
            add(scope_name, empty_label, _metrics(groups[empty_label]))
            return
        print(f"\n  --- By {scope_name} ---")
        for g in sorted(groups, key=lambda g: (g == empty_label, str(g))):
            m = _metrics(groups[g])
            _print_row(str(g), m); add(scope_name, str(g), m)
 
    group_block("category", "CATEGORY")
    group_block("sentiment", "SENTIMENT")
 
    print("\n  Legend: faith=faithfulness, rel=answer relevancy, "
          "anti=anti-hallucination (from refusals).")
    print("  '-' = no sample for that metric in the group.\n")
    return summary
 
 
# ----------------------------------------------------------------------
# Simpan CSV detail + ringkasan
# ----------------------------------------------------------------------
def _save_detail(rows):
    """Tulis baris detail per-interaksi ke DETAIL_CSV."""
    keys = ["key", "timestamp", "question", "category", "sentiment",
            "not_found", "faithfulness", "answer_relevancy", "antihalusinasi_ok"]
    with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
    print(f"[saved] {DETAIL_CSV}  ({len(rows)} rows)")
 
 
def _save_summary(summary):
    """Tulis baris ringkasan agregat ke SUMMARY_CSV."""
    keys = ["scope", "group", "n", "answered", "refused",
            "faithfulness", "answer_relevancy", "anti_hallucination", "generated_at"]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"[saved] {SUMMARY_CSV}  ({len(summary)} groups)")
 
 
# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Incremental RAG logbook evaluator")
    ap.add_argument("--logbook", default="rag_logbook.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge", choices=["gemini", "groq"], default="groq",
                    help="LLM judge backend (default groq, matches batch_run; gemini free tier is very limited)")
    ap.add_argument("--force", action="store_true",
                    help="re-score everything, ignore the cache in logbook_eval.csv")
    ap.add_argument("--ragas", action="store_true")
    args = ap.parse_args()
 
    if not Path(args.logbook).exists():
        raise SystemExit(f"Logbook '{args.logbook}' not found. Run app.py / batch_run.py first.")
 
    records = load_logbook(args.logbook, args.limit)
    if not records:
        raise SystemExit("Logbook is empty.")
 
    key_env = "GEMINI_API_KEY" if args.judge == "gemini" else "GROQ_API_KEY"
    if key_env not in os.environ:
        raise SystemExit(f"Set {key_env} first. Example (cmd): set {key_env}=...")
 
    if args.ragas:
        run_ragas(records, args.judge)
    else:
        run_judge(records, args.judge, force=args.force)