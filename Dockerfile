FROM python:3.10-slim

# Evitar preguntas interactivas durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias del sistema + compilar Python packages + limpiar
RUN apt-get update && \
    # Instalar dependencias de runtime
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libglib2.0-0 \
        wget && \
    # Instalar dependencias de compilación temporalmente
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        python3-dev && \
    # Instalar paquetes Python
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Limpiar dependencias de compilación para reducir tamaño
    apt-get purge -y --auto-remove \
        gcc \
        g++ \
        make \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copiar código de la aplicación
COPY kyc_processor.py .
COPY api_kyc.py .

# Crear directorio para modelos
RUN mkdir -p models

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Comando para ejecutar la API
CMD ["uvicorn", "api_kyc:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]