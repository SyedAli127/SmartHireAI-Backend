# ══════════════════════════════════════════════════════════════════════════════
# SmartHireAI — Campaign API (campaign_api.py + full pipeline)
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
# gcc      : some pip packages compile C extensions
# libpq-dev: psycopg2 (PostgreSQL driver)
# poppler-utils: pdf2image fallback (pdfplumber/pymupdf used first, but kept for safety)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY campaign_api.py .
COPY main_v2.py .
COPY database        database
COPY Shared          Shared
COPY services        services
COPY Parser          Parser
COPY Pre-Screening   Pre-Screening
COPY Interview_Agent Interview_Agent
COPY Create_JD       Create_JD
COPY Generate_Ad     Generate_Ad
COPY Post_Ad         Post_Ad
COPY Offer_Letter    Offer_Letter
COPY Communication_Scheduling Communication_Scheduling
COPY Ad_Performance  Ad_Performance
COPY Resume_Parser   Resume_Parser
COPY Pipeline        Pipeline
COPY Orchestrator    Orchestrator
COPY MCP             MCP

# Runtime directories (created empty; mount volumes in production)
RUN mkdir -p Generated_Ads uploads/resumes .runtime/campaign_runs secret

# ── Config ─────────────────────────────────────────────────────────────────────
# Pass all secrets as environment variables at runtime — do NOT bake .env into
# the image. Example:
#   docker run --env-file .env -p 8001:8001 smarthireai
EXPOSE 8001

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health').read()"

# ── Entrypoint ─────────────────────────────────────────────────────────────────
CMD ["uvicorn", "campaign_api:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
