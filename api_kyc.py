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
    mp4_path = os.path.join(TMP_DIR, "temp_video.mp4")
    ffmpeg_bin = "ffmpeg"
    subprocess.run([
        ffmpeg_bin, "-y", "-i", video_path,
        "-c:v", "libx264", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        mp4_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mp4_path

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

        # Convertir video y extraer frames
        video_mp4_path = convert_video_to_mp4(video_path)
        frames = extract_frames_from_video(video_mp4_path)

        # Procesar frames (sin Whisper)
        resultado = procesar_frames(frames, carnet_path, video_path=video_mp4_path, audio_wav_path=None)

        return JSONResponse(content=resultado)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
