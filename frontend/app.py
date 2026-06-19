"""
================================================================================
News Intelligence Assistant - Frontend (Streamlit)
================================================================================
Frontend murni tampilan: mesin RAG TIDAK jalan di sini. Frontend cuma memanggil
backend FastAPI lewat HTTP (POST /ask). Karena itu frontend ini ringan - gak load
model, gak butuh GPU, gak import torch/chromadb sama sekali.

Alur:
    [Streamlit UI] --POST /ask--> [FastAPI backend] --> [RAG: classify/retrieve/generate]
                   <--- JSON ----

Yang dibutuhkan biar jalan:
    1. Backend harus sudah nyala dulu (uvicorn main:app di folder backend/).
    2. API key (GROQ/GEMINI) di-set di environment BACKEND, bukan di sini.

Alamat backend diatur lewat env var BACKEND_URL (default http://localhost:8000).

JALANIN (dari folder frontend/):
    streamlit run app.py
================================================================================
"""

import os
import requests
import streamlit as st

# ============================================================
# KONFIGURASI
# ============================================================

# Alamat backend. Saat lokal default localhost; saat Docker di-set ke nama
# service backend (mis. http://backend:8000) lewat env var.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
ASK_URL = f"{BACKEND_URL}/ask"
HEALTH_URL = f"{BACKEND_URL}/health"

CATEGORIES = ["finance", "health", "inet", "news", "sport"]
SENTIMENTS = ["positif", "netral", "negatif"]
DATA_PERIODE = "1 Mei - 29 Mei 2026"

CONTOH_PERTANYAAN = [
    "Apa yang terjadi dengan IHSG belakangan ini?",
    "Berita teknologi terbaru apa saja?",
    "Kabar olahraga Indonesia yang membanggakan?",
    "Bagaimana sentimen berita kesehatan?",
]

st.set_page_config(page_title="News Intelligence Assistant", page_icon="📰", layout="wide")


# ============================================================
# GAYA / TEMA (CSS) - Dark Theme
# ============================================================

st.markdown(
    """
    <style>
    :root {
      --bg:#0B1120; --surface:#0F172A; --surface-2:#111C2E; --surface-3:#162033;
      --line:#2C3A4F; --line-soft:#223047; --text:#E5E7EB; --muted:#94A3B8;
      --muted-2:#CBD5E1; --brand:#38BDF8; --brand-2:#0EA5E9; --accent:#7DD3FC;
      --pos:#4ADE80; --net:#A8B0BD; --neg:#F87171; --warn:#FBBF24;
    }
    .stApp, [data-testid="stAppViewContainer"] {
      background:radial-gradient(circle at top left, rgba(56,189,248,.12), transparent 30%), var(--bg) !important;
      color:var(--text) !important;
    }
    [data-testid="stHeader"] { background:rgba(11,17,32,.78) !important; backdrop-filter:blur(10px); }
    .block-container { padding-top:1.3rem; padding-bottom:2rem; max-width:1080px; }
    html, body, [class*="css"] { font-family:"Segoe UI", system-ui, -apple-system, sans-serif; color:var(--text); }
    a { color:var(--accent); }

    .hero {
      background:linear-gradient(120deg, #111827 0%, #0B3A5E 52%, #075985 100%);
      border:1px solid rgba(125,211,252,.24);
      border-radius:18px; padding:22px 26px; color:#F8FAFC;
      box-shadow:0 14px 34px rgba(0,0,0,.35); margin-bottom:16px;
    }
    .hero-title { font-size:1.75rem; font-weight:800; letter-spacing:-.5px; line-height:1.15; }
    .hero-sub { font-size:.98rem; color:#D6E4F0; margin-top:6px; }
    .hero-badges { margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }
    .hero-pill {
      background:rgba(15,23,42,.55); border:1px solid rgba(203,213,225,.20);
      padding:6px 13px; border-radius:999px; font-size:.82rem; font-weight:600;
      backdrop-filter:blur(3px); color:#E2E8F0;
    }
    .hero-pill.period {
      background:rgba(56,189,248,.16); color:#E0F2FE;
      border:1px solid rgba(125,211,252,.35); font-weight:800;
    }

    .chip { display:inline-block; padding:2px 10px; border-radius:999px;
            font-size:.74rem; font-weight:800; margin-right:6px; border:1px solid transparent; }
    .chip-cat { background:rgba(56,189,248,.13); color:#BAE6FD; border-color:rgba(56,189,248,.22); }
    .chip-pos { background:rgba(74,222,128,.13); color:var(--pos); border-color:rgba(74,222,128,.22); }
    .chip-net { background:rgba(168,176,189,.14); color:#D1D5DB; border-color:rgba(168,176,189,.18); }
    .chip-neg { background:rgba(248,113,113,.13); color:var(--neg); border-color:rgba(248,113,113,.22); }

    .route { background:rgba(17,28,46,.92); border:1px solid var(--line); border-radius:12px;
             padding:11px 15px; margin:2px 0 16px; font-size:.9rem; color:var(--muted-2); }
    .route b { color:#E0F2FE; }

    .sec-title { font-size:1.06rem; font-weight:800; color:#E0F2FE; margin:4px 0 9px; }

    .sent-card { background:rgba(17,28,46,.95); border:1px solid var(--line); border-radius:14px;
                 padding:15px 16px; box-shadow:0 10px 24px rgba(0,0,0,.18); }
    .sbar { display:flex; height:15px; border-radius:8px; overflow:hidden;
            border:1px solid var(--line); background:#0F172A; }
    .sbar span { display:block; height:100%; }
    .sent-chips { margin-top:12px; line-height:2; }
    .sent-total { margin-top:8px; color:var(--muted); font-size:.78rem; }

    .src-card { display:flex; gap:12px; background:rgba(17,28,46,.95); border:1px solid var(--line);
                border-left:4px solid var(--brand-2); border-radius:12px; padding:12px 14px;
                margin-bottom:10px; box-shadow:0 10px 24px rgba(0,0,0,.16); transition:.15s; }
    .src-card:hover { box-shadow:0 16px 32px rgba(0,0,0,.28); transform:translateY(-1px); border-color:#3B4A63; }
    .src-num { font-weight:800; color:var(--accent); font-size:.95rem; min-width:24px; }
    .src-title { font-weight:750; color:#E0F2FE; text-decoration:none; font-size:.95rem; }
    .src-title:hover { color:#7DD3FC; text-decoration:underline; }
    .src-meta { margin-top:7px; font-size:.78rem; }
    .src-dist { color:var(--muted); font-size:.76rem; }

    .nf-card { display:flex; gap:14px; background:rgba(127,29,29,.22); border:1px solid rgba(248,113,113,.35);
               border-radius:14px; padding:16px 18px; }
    .nf-icon { font-size:1.6rem; }
    .nf-title { font-weight:800; color:#FCA5A5; font-size:1.02rem; }
    .nf-body { color:var(--text); margin-top:4px; font-size:.95rem; }
    .nf-note { color:var(--muted-2); margin-top:10px; font-size:.8rem;
               border-top:1px dashed rgba(248,113,113,.35); padding-top:8px; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
      border-radius:14px; border-color:var(--line)!important;
      box-shadow:0 10px 26px rgba(0,0,0,.20); background:rgba(17,28,46,.95) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] * { color:var(--text); }

    .stButton>button { border-radius:10px; font-weight:750; border:1px solid var(--line);
                       background:#111827; color:var(--text); }
    .stButton>button:hover { border-color:var(--brand); color:#E0F2FE; background:#152238; }
    .stButton>button[kind="primary"] { background:linear-gradient(120deg, #0284C7, #0EA5E9); border:none; color:#fff; }
    .stButton>button[kind="primary"]:hover { background:linear-gradient(120deg, #0369A1, #0284C7); color:#fff; }

    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
      background:#0F172A !important; border-right:1px solid var(--line-soft);
    }
    section[data-testid="stSidebar"] * { color:var(--text); }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color:#E0F2FE !important; }
    .period-note { background:rgba(251,191,36,.12); border:1px solid rgba(251,191,36,.32); color:#FDE68A;
                   border-radius:10px; padding:10px 12px; font-size:.82rem; line-height:1.45; }
    .period-note, .period-note * { color:#FDE68A !important; }
    .status-ok { background:rgba(74,222,128,.12); border:1px solid rgba(74,222,128,.32);
                 color:#86EFAC; border-radius:10px; padding:9px 12px; font-size:.82rem; }
    .status-bad { background:rgba(248,113,113,.12); border:1px solid rgba(248,113,113,.35);
                  color:#FCA5A5; border-radius:10px; padding:9px 12px; font-size:.82rem; line-height:1.45; }

    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, label { color:var(--muted-2) !important; }
    .stCaption, [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * { color:var(--muted) !important; }
    input, textarea {
      color:var(--text) !important; background:#111827 !important;
      caret-color:var(--accent) !important;
    }
    input::placeholder, textarea::placeholder { color:#64748B !important; opacity:1; }
    [data-baseweb="input"], [data-baseweb="base-input"],
    [data-baseweb="select"] > div {
      background:#111827 !important; border-color:var(--line) !important; color:var(--text) !important;
    }
    [data-baseweb="select"] * { color:var(--text) !important; }
    [data-baseweb="popover"], [role="listbox"] { background:#111827 !important; color:var(--text) !important; }
    [role="option"] { background:#111827 !important; color:var(--text) !important; }
    [role="option"]:hover { background:#1E293B !important; }
    [data-testid="stSlider"] * { color:var(--muted-2) !important; }
    hr { border-color:var(--line-soft) !important; }

    [data-testid="stAlert"] { background:rgba(251,191,36,.12) !important; color:var(--text) !important; border-color:rgba(251,191,36,.35) !important; }

    .foot { color:var(--muted); font-size:.78rem; text-align:center; margin-top:26px;
            border-top:1px solid var(--line); padding-top:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PANGGILAN KE BACKEND
# ============================================================

def check_backend():
    """Cek apakah backend hidup & komponen siap. Return (ok, pesan)."""
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        if r.status_code == 200 and r.json().get("components_ready"):
            return True, "Backend tersambung, komponen siap."
        return False, f"Backend menjawab tapi belum siap (status {r.status_code})."
    except requests.exceptions.RequestException:
        return False, "Backend tidak terjangkau."


def ask_backend(question, provider, k, force_category, sentiment_filter):
    """Kirim pertanyaan ke backend (POST /ask) dan kembalikan dict jawaban.

    Bentuk balikannya mengikuti output rag_answer di backend:
    answer, sources, kategori_query, kategori_conf, filter, sentimen.
    """
    payload = {
        "question": question,
        "provider": provider,
        "k": k,
        "force_category": force_category,
        "sentiment_filter": sentiment_filter,
    }
    r = requests.post(ASK_URL, json=payload, timeout=120)
    if r.status_code != 200:
        # Coba ambil detail error dari backend.
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"Backend error {r.status_code}: {detail}")
    return r.json()


# ============================================================
# HELPER TAMPILAN
# ============================================================

_SENT_KIND = {"positif": "pos", "netral": "net", "negatif": "neg"}
_SENT_COLOR = {"positif": "#3F9A4E", "netral": "#8A8F98", "negatif": "#C0392B"}


def cat_chip(kategori):
    return f'<span class="chip chip-cat">{kategori}</span>'


def sent_chip(sentiment):
    kind = _SENT_KIND.get(sentiment, "net")
    return f'<span class="chip chip-{kind}">{sentiment}</span>'


def filter_to_text(filt):
    if not filt:
        return "tidak ada"
    parts = []
    if "kategori" in filt:
        parts.append(f"kategori = {filt['kategori']}")
    if "sentiment" in filt:
        parts.append(f"sentimen = {filt['sentiment']}")
    if "fallback" in filt:
        parts.append(filt["fallback"])
    return " · ".join(parts) if parts else "tidak ada"


def is_not_found(out):
    if not out["sources"]:
        return True
    low = (out["answer"] or "").lower()
    markers = ["tidak menemukan", "tidak ada informasi", "tidak ditemukan",
               "tidak tersedia", "tidak ada artikel relevan"]
    return any(m in low for m in markers)


def render_sentiment(sentimen):
    total = sum(sentimen.values())
    segs = ""
    for s in SENTIMENTS:
        n = sentimen.get(s, 0)
        pct = (n / total * 100) if total else 0
        segs += f'<span style="width:{pct:.1f}%;background:{_SENT_COLOR[s]}"></span>'
    chips = ""
    for s in SENTIMENTS:
        chips += f'{sent_chip(s)}<b style="color:{_SENT_COLOR[s]}">{sentimen.get(s, 0)}</b>&nbsp;&nbsp;'
    return (
        '<div class="sent-card">'
        '<div class="sec-title">📊 Sentimen Sumber</div>'
        f'<div class="sbar">{segs}</div>'
        f'<div class="sent-chips">{chips}</div>'
        f'<div class="sent-total">Total {total} sumber terambil</div>'
        '</div>'
    )


def render_sources(sources):
    cards = ""
    for i, s in enumerate(sources, 1):
        cards += (
            '<div class="src-card">'
            f'<div class="src-num">[{i}]</div>'
            '<div style="flex:1">'
            f'<a class="src-title" href="{s["url"]}" target="_blank">{s["judul"]}</a>'
            f'<div class="src-meta">{cat_chip(s["kategori"])}{sent_chip(s["sentiment"])}'
            f'<span class="src-dist">jarak {s["distance"]}</span></div>'
            '</div></div>'
        )
    return cards


# ============================================================
# ANTARMUKA PENGGUNA (UI)
# ============================================================

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">📰 News Intelligence Assistant</div>
      <div class="hero-sub">Tanya-jawab berita Indonesia berbasis RAG · klasifikasi kategori · analisis sentimen</div>
      <div class="hero-badges">
        <span class="hero-pill period">📅 Periode data: {DATA_PERIODE}</span>
        <span class="hero-pill">🗂️ 5 kategori</span>
        <span class="hero-pill">🧠 IndoBERT + LLM</span>
        <span class="hero-pill">🔒 Grounded · anti-halusinasi</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Pengaturan")

    # Status koneksi backend.
    ok, msg = check_backend()
    if ok:
        st.markdown(f'<div class="status-ok">🟢 {msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="status-bad">🔴 {msg}<br>Pastikan backend nyala: '
            f'<code>uvicorn main:app</code> di folder <code>backend/</code>.<br>'
            f'Target: <code>{BACKEND_URL}</code></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    provider = st.selectbox("LLM Provider", ["groq", "gemini"], index=0,
                            help="API key di-set di environment BACKEND, bukan di sini.")

    st.subheader("🔎 Filter")
    force_category = st.selectbox("Kategori", ["Auto"] + CATEGORIES, index=0,
                                  help="Auto = ditentukan otomatis dari pertanyaan (classifier)")
    sentiment_filter = st.selectbox("Sentimen", ["Semua"] + SENTIMENTS, index=0)
    k = st.slider("Jumlah sumber (k)", 3, 10, 5)

    st.divider()
    st.markdown(
        f'<div class="period-note">📅 <b>Cakupan data:</b> {DATA_PERIODE}.<br>'
        'Pertanyaan di luar rentang/topik ini akan dijawab jujur "tidak ditemukan".</div>',
        unsafe_allow_html=True,
    )
    st.caption("Frontend ini hanya menampilkan; seluruh proses RAG dijalankan oleh backend API.")


# --- Contoh pertanyaan ---
def _use_example(text):
    st.session_state["q"] = text


if "q" not in st.session_state:
    st.session_state["q"] = ""

query = st.text_input("💬 Pertanyaan kamu:", key="q",
                      placeholder="Contoh: Apa yang terjadi dengan IHSG belakangan ini?")
ask = st.button("Tanya", type="primary", use_container_width=False)

st.caption("Atau coba salah satu contoh:")
ex_cols = st.columns(len(CONTOH_PERTANYAAN))
for col, contoh in zip(ex_cols, CONTOH_PERTANYAAN):
    col.button(contoh, key=f"ex_{contoh[:12]}", on_click=_use_example, args=(contoh,),
               use_container_width=True)

# --- Pemrosesan ---
if ask:
    if not query.strip():
        st.warning("Tulis pertanyaan dulu.")
    else:
        try:
            with st.spinner("Mengirim ke backend & menyusun jawaban..."):
                out = ask_backend(query, provider, k, force_category, sentiment_filter)
        except requests.exceptions.ConnectionError:
            st.error(f"Tidak bisa terhubung ke backend di {BACKEND_URL}. "
                     "Pastikan backend (uvicorn) sudah nyala.")
            st.stop()
        except Exception as e:
            st.error(f"Terjadi error: {e}")
            st.stop()

        st.markdown("")

        if is_not_found(out):
            st.markdown(
                '<div class="nf-card"><div class="nf-icon">🔍</div><div>'
                '<div class="nf-title">Tidak ditemukan dalam data</div>'
                f'<div class="nf-body">{out["answer"]}</div>'
                f'<div class="nf-note">Cakupan data sistem: <b>{DATA_PERIODE}</b>. '
                'Pertanyaan di luar topik atau rentang waktu ini tidak dapat dijawab '
                'dari artikel yang tersedia.</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="route">📂 <b>Kategori terdeteksi:</b> {out["kategori_query"]} '
                f'({out["kategori_conf"]}) &nbsp;·&nbsp; 🔎 <b>Filter:</b> '
                f'{filter_to_text(out["filter"])}</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="sec-title">💬 Jawaban</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(out["answer"])

            st.markdown("")

            col_sent, col_src = st.columns([1, 2], gap="large")
            with col_sent:
                st.markdown(render_sentiment(out["sentimen"]), unsafe_allow_html=True)
            with col_src:
                st.markdown(f'<div class="sec-title">📰 Sumber ({len(out["sources"])})</div>',
                            unsafe_allow_html=True)
                st.markdown(render_sources(out["sources"]), unsafe_allow_html=True)

# --- Footer ---
st.markdown(
    f'<div class="foot">News Intelligence Assistant · basis pengetahuan periode {DATA_PERIODE} · '
    'RAG (IndoBERT + ChromaDB + LLM) · frontend Streamlit -> backend FastAPI</div>',
    unsafe_allow_html=True,
)