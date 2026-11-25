FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY component3 ./component3
RUN pip install --no-cache-dir ".[api]" && useradd --create-home app
USER app
EXPOSE 8000
CMD ["uvicorn", "component3.api:app", "--host", "0.0.0.0", "--port", "8000"]
