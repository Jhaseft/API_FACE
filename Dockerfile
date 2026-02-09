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

# Instalar todas las librerías del requirements.txt usando wheels precompilados
RUN pip install --no-cache-dir -r requirements.txt

# Directorio de la app
WORKDIR /app
COPY . .

# Exponer puerto
EXPOSE 8080

# Comando de inicio
CMD ["uvicorn", "api_kyc:app", "--host", "0.0.0.0", "--port", "8080"]
