FROM python:3.11-slim AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${VIRTUAL_ENV}"

COPY docker/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements-runtime.txt

FROM python:3.11-slim

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system llmrouter \
    && useradd --system --gid llmrouter --create-home --home-dir /home/llmrouter llmrouter

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
# Copy only what the app runs; API keys come from compose environment variables.
COPY --chown=llmrouter:llmrouter main.py /app/main.py
COPY --chown=llmrouter:llmrouter src /app/src
COPY --chown=llmrouter:llmrouter slack /app/slack
COPY --chown=llmrouter:llmrouter config /app/config

RUN mkdir -p /app/logs /app/data /app/models \
    && chown -R llmrouter:llmrouter /app

USER llmrouter

EXPOSE 8080

CMD ["python", "main.py", "start-api", "--config", "config/config.compose.yaml"]
