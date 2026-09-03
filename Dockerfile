FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN mkdir -p /data/documents
ENV DOCUMENT_ROOT=/data/documents SQLITE_PATH=/data/memory.db
EXPOSE 8080
# Healthcheck hits /healthz inside the container; PORT is respected so the
# check stays valid if the listen port is overridden.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"8080\")}/healthz', timeout=4)" ]
CMD ["python", "-m", "memory_store.server"]
