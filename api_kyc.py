# api_kyc.py (actualizado)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, tempfile, subprocess
import cv2
from kyc_processor import procesar_frames

app = FastAPI(title="KYC Processor API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TMP_DIR = tempfile.gettempdir()

def save_upload_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return destination

def convert_video_to_mp4(video_path: str):
    """
    Convierte un video a MP4 con H264/AAC.
    Si falla, devuelve el path original y video_convertido=False.
    """
    mp4_path = os.path.join(TMP_DIR, "temp_video.mp4")
    ffmpeg_bin = "ffmpeg"
    try:
        result = subprocess.run(
            [
                ffmpeg_bin, "-y", "-i", video_path,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                mp4_path
            ],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("[FFMPEG ERROR]", result.stderr)
            return video_path, False  # fallback: usamos .webm
        return mp4_path, True
    except Exception as e:
        print(f"[ERROR] convert_video_to_mp4: {e}")
        return video_path, False

def extract_frames_from_video(video_path, max_frames=30, frame_skip=5):
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
            small = cv2.resize(frame, (320, 240))
            frames.append(small)
        frame_count += 1
    cap.release()
    if not frames:
        raise RuntimeError("No se pudieron extraer frames del video")
    return frames

@app.get("/")
async def root():
    return {"message": "KYC Processor API está corriendo"}

@app.post("/registro-face/verify")
async def verify_kyc(
    carnet: UploadFile = File(...),
    video: UploadFile = File(...),
):
    try:
        # Guardar archivos
        carnet_path = save_upload_file(carnet, os.path.join(TMP_DIR, carnet.filename))
        video_path  = save_upload_file(video, os.path.join(TMP_DIR, video.filename))

        # Intentar convertir video a MP4
        video_mp4_path, converted = convert_video_to_mp4(video_path)

        # Extraer frames (si no se pudo convertir, se usa el WebM original)
        frames = extract_frames_from_video(video_mp4_path)

        # Procesar frames
        resultado = procesar_frames(frames, carnet_path, video_path=video_mp4_path)

        # Agregamos info de conversión
        resultado["video_convertido"] = converted
        if not converted:
            resultado["mensajes"].append(
                "⚠️ Formato de video no soportado o conversión fallida. Se usó el video original."
            )

        return JSONResponse(content=resultado)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
