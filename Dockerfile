FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_ENV=production
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
# uid 1000 because Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser
EXPOSE 7860
# One worker on purpose: jobs, datasets and quizzes live in this process's memory (see README).
# PORT is set by Render/Cloud Run; Hugging Face and Cloudflare use 7860
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
