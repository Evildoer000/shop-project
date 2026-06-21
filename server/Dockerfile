# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY requirements-docker.txt ./requirements-docker.txt

ARG INSTALL_MULTIMODAL=false
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_MULTIMODAL" = "true" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends build-essential patch \
        && rm -rf /var/lib/apt/lists/* \
        && python -m pip install --upgrade pip \
        && python -m pip install -r requirements.txt; \
    else \
        python -m pip install --upgrade pip \
        && python -m pip install -r requirements-docker.txt; \
    fi

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p /app/logs /data/uploads \
    && useradd --create-home --shell /bin/sh appuser \
    && chown -R appuser:appuser /app /data/uploads

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
