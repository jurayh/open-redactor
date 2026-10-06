FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg nodejs npm tesseract-ocr libglib2.0-0 \
    && npm install --prefix / @meta-sam/parser@0.0.13 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip \
    && python -m pip install "/app[pii,server]"

WORKDIR /work
ENTRYPOINT ["open-redactor"]
CMD ["--help"]
