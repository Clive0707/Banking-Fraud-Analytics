# Banking Fraud Analytics — PySpark local[*] + Flask dashboard
#
# NO-HADOOP DIRECTIVE: Spark runs in local standalone mode. No Hadoop, HDFS or
# YARN is installed or configured in this image.

FROM python:3.12-slim

# Spark needs a JVM; Temurin 17 is the version CI tests against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SPARK_DRIVER_MEMORY=4g \
    SPARK_EXECUTOR_MEMORY=4g

WORKDIR /app

# Dependencies first so the layer caches across source edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY run.py pytest.ini ./

# The raw CSV and generated artefacts are mounted at runtime rather than baked
# into the image:
#   docker run -p 5000:5000 \
#     -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" banking-fraud-analytics
RUN mkdir -p data/raw data/processed models reports

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=4).status==200 else 1)"

# Serves with waitress. Add --all to rebuild every artefact on start.
CMD ["python", "run.py", "--production", "--port", "5000"]
