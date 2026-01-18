FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.21 /uv /uvx /bin/

WORKDIR /app

ENV XDG_CACHE_HOME=/tmp/.cache \
    UV_CACHE_DIR=/tmp/.cache/uv \
    XDG_CONFIG_HOME=/tmp/.config \
    XDG_STATE_HOME=/tmp/.local/state \
    XDG_DATA_HOME=/tmp/.local/share

RUN mkdir -p /tmp/.cache/uv /tmp/.config /tmp/.local/state /tmp/.local/share

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen \
    && chmod -R a+rwX /tmp/.cache /tmp/.config /tmp/.local

COPY src/ ./src/
COPY data/ ./data/

CMD ["uv", "run", "marimo", "edit", "--host", "0.0.0.0", "--port", "3000", "--no-token"]
