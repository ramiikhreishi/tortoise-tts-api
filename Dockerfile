FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-venv \
    python3.9-distutils \
    python3.9-dev \
    build-essential \
    curl \
    git \
    ninja-build \
    libaio-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.9 — the generic get-pip.py bootstrap dropped support
# for EOL Python versions, so use the version-pinned one for 3.9 specifically.
RUN curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9

# Set python and pip aliases
RUN ln -s /usr/bin/python3.9 /usr/bin/python && ln -s /usr/local/bin/pip /usr/bin/pip


WORKDIR /app

COPY . .

RUN pip install  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install transformers

RUN pip install -e .

RUN pip install -r requirements_api.txt


ENV LOG_DIR=/app/logs

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
