FROM python:3.10-slim

# Evitar preguntas interactivas
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias del sistema + compilar Python packages + limpiar
RUN apt-get update && \
    # Runtime dependencies
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libglib2.0-0 \
        wget \
        curl && \
    # Build dependencies (temporal)
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        python3-dev && \
    # Upgrade pip e instalar paquetes Python
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Limpiar build dependencies
    apt-get purge -y --auto-remove \
        gcc \
        g++ \
        make \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Crear directorio para modelos
RUN mkdir -p /app/models

# ⭐ DESCARGAR MODELOS DURANTE EL BUILD (no al iniciar)
RUN echo "Descargando modelos OpenCV..." && \
    wget -q -O /app/models/deploy.prototxt \
        https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt && \
    wget -q -O /app/models/res10_300x300_ssd_iter_140000.caffemodel \
        https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel && \
    echo "✓ Modelos descargados correctamente" && \
    ls -lh /app/models/

# Copiar código de la aplicación
COPY kyc_processor.py .
COPY api_kyc.py .

# Exponer puerto
EXPOSE 8000

# Healthcheck simple usando curl (ya instalado)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para ejecutar la API con logging
CMD ["uvicorn", "api_kyc:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]