# Multi-stage: wheels are built once, the runtime image stays small.
FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir --upgrade pip wheel
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[postgis]"

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    S1CD_RUNS_DIR=/data/runs
# rasterio's bundled GDAL links against libexpat, which python:*-slim does not ship
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app && mkdir -p /data/runs && chown -R app:app /data
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
WORKDIR /home/app
COPY --chown=app:app app ./app
COPY --chown=app:app examples ./examples
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
