FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system libraries required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port 5000
EXPOSE 5000

# Start Flask with Gunicorn (1 worker + 2 threads for low memory usage)
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "2", "--bind", "0.0.0.0:5000", "--timeout", "120"]
