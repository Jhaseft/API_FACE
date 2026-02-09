FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TMPDIR=/tmp
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=2

# Dependencias del sistema necesarias para compilación y multimedia
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Crear entorno virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Actualizar pip y setuptools en el venv
RUN pip install --upgrade pip setuptools wheel

# Copiar requirements y luego instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la app
COPY . .

EXPOSE 8080

CMD ["uvicorn", "api_kyc:app", "--host", "0.0.0.0", "--port", "8080"]
