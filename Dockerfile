# Imagen base con Python 3.11 (estable para mediapipe/whisper)
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

# Crear entorno virtual
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Actualizar pip y asegurar tensorflow-cpu
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip uninstall -y tensorflow tensorflow-gpu \
    && pip install tensorflow-cpu \
    && pip install -r requirements.txt

# Copiar el código
COPY . .

# Exponer puerto
EXPOSE 8080

# Comando de inicio
CMD ["uvicorn", "api_kyc:app", "--host", "0.0.0.0", "--port", "8080"]
