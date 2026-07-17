FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy context
COPY . .
RUN uv pip install --system --no-cache -e "."

# Copy application code
COPY app/ ./app/

# Expose port (Cloud Run uses $PORT)
ENV PORT=8080
EXPOSE 8080

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
