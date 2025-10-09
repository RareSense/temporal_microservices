# --- multi-service build context ---
ARG PY_VER=3.11
FROM python:${PY_VER}-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# ---------- jewellery classifier layer ----------
FROM base AS jewelry-classify
COPY jewelry_classifier/ ./jewelry_classifier
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["gunicorn","-c","jewelry_classifier/gunicorn_conf.py","jewelry_classifier.app:app"]
