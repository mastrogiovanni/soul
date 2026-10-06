FROM python:3.12-slim AS controller
COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --no-cache \
    && useradd --create-home soul \
    && mkdir -p /data \
    && chown soul:soul /data
USER soul
EXPOSE 8000
ENTRYPOINT ["soul"]
CMD ["controller", "--host", "0.0.0.0", "--state-dir", "/data"]
