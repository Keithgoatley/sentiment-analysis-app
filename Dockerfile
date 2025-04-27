# Dockerfile
FROM python:3.12-slim

# 1) System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# 2) Set workdir
WORKDIR /app

# 3) Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy rest of code
COPY . .

# 5) Expose Streamlit port
EXPOSE 8501

# 6) Entrypoint
ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.headless=true"]
