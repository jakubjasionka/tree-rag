FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TREE_RAG_BASE_DIR=/data/treerag

COPY pyproject.toml README.md LICENSE ./
COPY scoped_memory ./scoped_memory
COPY examples ./examples

RUN pip install --no-cache-dir -e ".[http]" && mkdir -p /data/treerag

VOLUME ["/data/treerag"]

ENTRYPOINT ["python", "examples/cli.py", "--base-dir", "/data/treerag"]
CMD ["health"]
