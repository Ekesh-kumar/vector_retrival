# Use an official Python image as base
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency list first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose FastAPI port
RUN chmod +x ./start.sh

EXPOSE 8000

CMD ["./start.sh"]

