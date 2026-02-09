# Imagen base con Python 3.10 (estable para mediapipe/whisper)
FROM python:3.11-slim

# Evitar preguntas interactivas
ENV DEBIAN_FRONTEND=noninteractive

# Forzar TensorFlow a CPU y suprimir logs
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TMPDIR=/tmp

# Dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Directorio de la app
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copiar código
COPY . .

# Exponer puerto
EXPOSE 8080

# Comando de inicio
CMD ["uvicorn", "api_kyc:app", "--host", "0.0.0.0", "--port", "8080"]
