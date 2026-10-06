FROM python:3.12-slim AS controller
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home soul && mkdir -p /data && chown soul:soul /data
USER soul
EXPOSE 8000
ENTRYPOINT ["soul"]
CMD ["controller", "--host", "0.0.0.0", "--state-dir", "/data"]
