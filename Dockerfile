FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (the `app` package, including knowledge_base.md).
COPY app ./app

EXPOSE 8000

# Fail the container's health if the app stops serving /health. Honours $PORT
# (platforms like Render inject it) and falls back to 8000 for local/VM runs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health').status==200 else 1)"

# Shell form so ${PORT} is expanded at runtime; exec keeps uvicorn as PID 1 so
# it receives shutdown signals. Render sets $PORT; elsewhere it defaults to 8000.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
