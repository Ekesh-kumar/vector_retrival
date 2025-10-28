#!/bin/sh
set -e

echo "Waiting for PostgreSQL to start..."
until nc -z db 5432; do
  sleep 1
done

echo "Running Alembic migrations..."
alembic upgrade head

echo "Starting FastAPI..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
