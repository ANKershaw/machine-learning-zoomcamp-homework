FROM agrigorev/zoomcamp-model:2025
LABEL authors="ANkershaw"

#RUN pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" ./

RUN uv sync

COPY "predict.py" "pipeline_v1.bin" ./

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]