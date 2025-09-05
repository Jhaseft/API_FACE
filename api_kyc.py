# api_kyc.py (mejorado)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, tempfile, subprocess
import cv2
from kyc_processor import procesar_frames

app = FastAPI(title="KYC Processor API")

# =========================
# CORS
# =========================
origins = [
    "https://transfers.elchangarrodelima.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TMP_DIR = tempfile.gettempdir()

# =========================
# Utilidades
# =========================
def save_upload_file(upload_file: UploadFile) -> str:
    """Guarda un archivo subido en un archivo temporal y devuelve su path"""
    ext = os.path.splitext(upload_file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=TMP_DIR) as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name

def convert_video_to_mp4(video_path: str):
    """
    Convierte un video a MP4 con H264/AAC.
    Retorna (path_del_video, video_convertido:bool)
    """
    mp4_path = os.path.join(TMP_DIR, "temp_video.mp4")
    ffmpeg_bin = "ffmpeg"
    try:
        result = subprocess.run(
            [
                ffmpeg_bin, "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                mp4_path
            ],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("[FFMPEG ERROR]", result.stderr)
            return video_path, False
        return mp4_path, True
    except Exception as e:
        print(f"[ERROR] convert_video_to_mp4: {e}")
        return video_path, False

def extract_frames_from_video(video_path, max_frames=30, frame_skip=5):
    """
    Extrae frames del video. Redimensiona para acelerar procesamiento.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video para extracción de frames")
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0 and len(frames) < max_frames:
            # Redimensiona manteniendo proporción
            h, w = frame.shape[:2]
            scale = 320 / w if w > h else 240 / h
            new_w, new_h = int(w*scale), int(h*scale)
            small = cv2.resize(frame, (new_w, new_h))
            frames.append(small)
        frame_count += 1
    cap.release()
    if not frames:
        raise RuntimeError("No se pudieron extraer frames del video")
    return frames

# =========================
# Endpoints
# =========================
@app.get("/")
async def root():
    return {"message": "KYC Processor API está corriendo"}

@app.post("/registro-face/verify")
async def verify_kyc(
    carnet: UploadFile = File(...),
    video: UploadFile = File(...),
):
    try:
        # Guardar archivos temporales
        carnet_path = save_upload_file(carnet)
        video_path  = save_upload_file(video)

        # Convertir video a MP4 si es posible
        video_mp4_path, converted = convert_video_to_mp4(video_path)

        # Extraer frames (si falla conversión, usa original)
        frames = extract_frames_from_video(video_mp4_path)

        # Procesar KYC
        resultado = procesar_frames(frames, carnet_path, video_path=video_mp4_path)

        # Información de conversión
        resultado["video_convertido"] = converted
        if not converted:
            resultado["mensajes"].append(
                "⚠️ Formato de video no soportado o conversión fallida. Se usó el video original."
            )

        return JSONResponse(content=resultado)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e), "mensajes": ["Ocurrió un error al procesar el KYC"]},
            status_code=500
        )
