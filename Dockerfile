FROM python:3.11-slim

# Evitar preguntas interactivas
ENV DEBIAN_FRONTEND=noninteractive

# Actualizar e instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear virtualenv
RUN python -m venv /opt/venv

# Activar virtualenv y actualizar pip/setuptools
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Copiar tu código
WORKDIR /app
COPY . /app

# Instalar dependencias de tu proyecto
RUN /opt/venv/bin/pip install -r requirements.txt

# Comando por defecto
CMD ["/opt/venv/bin/uvicorn", "api_kyc:app", "--host", "0.0.0.0", "--port", "8080"]
