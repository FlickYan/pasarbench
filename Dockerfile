# The PasarBench agent service (docs/SERVICE.md).
#
#   docker build -t pasarbench-service .
#   docker compose -f deploy/compose.yaml up --build     PostgreSQL, the migration, the API
#
# One stage: every dependency ships as a wheel for this Python, so no compiler.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements-service.txt .
RUN pip install -r requirements-service.txt

COPY alembic.ini .
COPY pasarbench/ pasarbench/

# Not root. data/ is where a SQLite database lives when no PostgreSQL is given.
RUN useradd --system --uid 10001 --no-create-home pasar \
    && mkdir -p /app/data && chown pasar /app/data
USER pasar

ENV PASAR_DATABASE_URL=sqlite:////app/data/pasar.db
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["python", "-m", "pasarbench.service", "serve", "--host", "0.0.0.0", "--port", "8000"]
