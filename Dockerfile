FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY . .

RUN uv venv && uv pip install --no-cache --allow-insecure-host pypi.org --allow-insecure-host files.pythonhosted.org . dwave-neal

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# /app      → makes `src` importable as a package (for relative imports in app.py)
# /app/src  → makes `util` importable as a top-level module
ENV PYTHONPATH=/app:/app/src

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
