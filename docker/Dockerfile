FROM python:3.14.7-slim-trixie@sha256:51dafde81dbdb6ebde285137a295cf18a47ca95234fe388a343719cb97305b3d

ARG RCLONE_VERSION=v1.75.2
ARG RCLONE_SHA256=349ac8fba6ff65d6247043f1750cdcb518ec5d500ef91463a10d37c0ccdf3702

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL -o /tmp/rclone.zip "https://downloads.rclone.org/${RCLONE_VERSION}/rclone-${RCLONE_VERSION}-linux-amd64.zip" \
    && echo "${RCLONE_SHA256}  /tmp/rclone.zip" | sha256sum -c - \
    && unzip -j /tmp/rclone.zip "*/rclone" -d /usr/local/bin \
    && chmod 0755 /usr/local/bin/rclone \
    && rm /tmp/rclone.zip

WORKDIR /app

COPY docker/requirements.lock /tmp/requirements.lock
RUN pip install \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        --requirement /tmp/requirements.lock \
    && rm /tmp/requirements.lock

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
RUN pip install --no-deps --no-build-isolation --editable .

RUN rclone version \
    && python -c "import torch; from scout.tokenizer.frozen import load_tokenizer; load_tokenizer(); print(torch.__version__)"

CMD ["bash"]
