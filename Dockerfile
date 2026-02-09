# Imagen base Python 3.10
FROM python:3.10-slim

# Evitar preguntas interactivas
ENV DEBIAN_FRONTEND=noninteractive
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
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Crear y activar entorno virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Actualizar pip
RUN pip install --upgrade pip

# Copiar requirements
COPY requirements.txt .

# Instalar librerías principales usando wheels (sin compilar desde cero)
RUN pip install numpy opencv-python-headless mediapipe

# Instalar el resto de librerías normalmente
RUN pip install fastapi==0.112.0 \
    uvicorn[standard]==0.23.2 \
    python-multipart==0.0.6 \
    deepface[torch]==0.0.89 \
    soundfile==0.12.1 \
    webrtcvad==2.0.10 \
    ffmpeg-python==0.1.18 \
    tensorflow==2.15.0 \
    keras==2.15.0 \
    typing-extensions>=4.8.0

# Directorio de la app
WORKDIR /app
COPY . .

# Exponer puerto
EXPOSE 8080

# Comando de inicio
CMD ["uvicorn", "api_kyc:app", "--host", "0.0.0.0", "--port", "8080"]
