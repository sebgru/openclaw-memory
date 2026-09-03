FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN mkdir -p /data/documents
ENV DOCUMENT_ROOT=/data/documents SQLITE_PATH=/data/memory.db
EXPOSE 8080
CMD ["python", "-m", "memory_store.server"]
