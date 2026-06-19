# ============================================================
# Dockerfile - Frontend (Streamlit)
# ============================================================
# Image buat layanan frontend. Ringan: cuma butuh Streamlit + requests,
# GAK ada torch/chromadb/model di sini (semua proses RAG ada di backend).
# Di-build lewat docker-compose (service: frontend).
# ============================================================

# Base image Python versi slim (kecil, tanpa paket OS yang gak perlu).
FROM python:3.11-slim

# Semua perintah & path relatif setelah ini dijalankan dari /app di dalam container.
WORKDIR /app

# Salin daftar dependency DULU (sebelum kode), biar layer hasil pip install
# bisa di-cache Docker dan gak ngulang install tiap kali app.py berubah.
COPY requirements.txt .

# Install dependency. --no-cache-dir biar image gak kebawa cache pip (lebih kecil).
RUN pip install --no-cache-dir -r requirements.txt

# Baru salin kode aplikasinya.
COPY app.py ./

# Tandai port yang dipakai Streamlit (dokumentasi; mapping port asli di compose).
EXPOSE 8501

# Jalankan Streamlit saat container start.
#   --server.port=8501      -> samain dengan port yang di-expose
#   --server.address=0.0.0.0 -> dengar dari semua interface biar bisa diakses
#                               dari luar container (bukan cuma localhost internal)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]