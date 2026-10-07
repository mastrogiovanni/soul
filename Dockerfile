FROM python:3.12-slim AS controller
COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"
ENV CONTEXT_ROOT=/soul HF_HOME=/data/huggingface PYTHONDONTWRITEBYTECODE=1
COPY pyproject.toml uv.lock .python-version README.md install.sh ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --no-cache \
    && groupadd --gid 1000 soul \
    && useradd --uid 1000 --gid 1000 --create-home soul \
    && mkdir -p /data \
    && chown 1000:1000 /data
USER 1000:1000
EXPOSE 8000
ENTRYPOINT ["soul"]
CMD ["controller", "--host", "0.0.0.0", "--state-dir", "/data"]
