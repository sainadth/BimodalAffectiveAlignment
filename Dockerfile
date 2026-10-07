# Remote study server: FastAPI + models + built React frontend. CPU-only.
FROM node:20-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false HF_HOME=/models
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# CPU torch wheels keep the image ~2.5 GB instead of ~7 GB
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -e ".[eval]" fastapi "uvicorn[standard]"
COPY server ./server
COPY checkpoints/fer2013_finetune.pt checkpoints/face_detection_yunet_2023mar.onnx ./checkpoints/
COPY --from=web /web/dist ./web/dist
VOLUME ["/app/data", "/models"]
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
