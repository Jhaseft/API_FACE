# Imagen base con Python 3.10 (más estable para mediapipe/whisper)
FROM python:3.10-slim

# Evitar preguntas interactivas al instalar paquetes
ENV DEBIAN_FRONTEND=noninteractive
# Forzar TensorFlow a usar CPU y suprimir warnings
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TMPDIR=/tmp

# Instalar dependencias del sistema necesarias para OpenCV y FFmpeg
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de la app
WORKDIR /app

# Copiar requirements y luego instalar
COPY requirements.txt .

# Crear un virtualenv dentro del contenedor
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar pip y dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copiar el código de la app
COPY . .

# Exponer puerto 8080 para Railway
EXPOSE 8080

# Comando de inicio
CMD ["uvicorn", "api_kyc:app", "--host", "0.0.0.0", "--port", "8080"]
