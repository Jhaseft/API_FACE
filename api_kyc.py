# api_kyc.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, tempfile, subprocess
import cv2
# FIX EXIF: corrige rotación de fotos tomadas con celular (portrait/landscape)
# Para desactivar: comentar el import y la llamada a fix_image_orientation() en verify_kyc
from PIL import Image, ImageOps
from kyc_processor import procesar_frames, select_best_frame, compare_faces_external

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

# Score mínimo de liveness para considerar la prueba válida
LIVENESS_MIN_SCORE = 35.0

# =========================
# Utilidades
# =========================
# FIX EXIF: aplica la rotación embebida en el JPEG (fotos de celular en cualquier orientación)
# Para desactivar: comentar el cuerpo de la función y dejar solo "pass"
def fix_image_orientation(image_path: str) -> None:
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)  # rota los píxeles según el tag EXIF
        img.save(image_path)
    except Exception as e:
        print(f"[WARN] fix_image_orientation: {e}")  # no rompe el flujo si falla


def save_upload_file(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=TMP_DIR) as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name


def convert_video_to_mp4(video_path: str):
    mp4_path = os.path.join(TMP_DIR, "temp_video.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                mp4_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("[FFMPEG ERROR]", result.stderr)
            return video_path, False
        return mp4_path, True
    except Exception as e:
        print(f"[ERROR] convert_video_to_mp4: {e}")
        return video_path, False


def extract_frames_from_video(video_path, max_frames=30, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video para extracción de frames")
    frames, frame_count = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0 and len(frames) < max_frames:
            h, w  = frame.shape[:2]
            scale = 320 / w if w > h else 240 / h
            frames.append(cv2.resize(frame, (int(w * scale), int(h * scale))))
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
    video:  UploadFile = File(...),
):
    carnet_path     = None
    video_path      = None
    best_frame_path = None

    try:
        # 1. Guardar archivos subidos
        carnet_path = save_upload_file(carnet)
        fix_image_orientation(carnet_path)  # FIX EXIF: corrige rotación antes de enviar al servicio
        video_path  = save_upload_file(video)

        # 2. Convertir video a MP4
        video_mp4_path, converted = convert_video_to_mp4(video_path)

        # 3. Extraer frames
        frames = extract_frames_from_video(video_mp4_path)

        # 4. Liveness: movimiento, parpadeo, audio
        liveness = procesar_frames(frames, video_path=video_mp4_path)

        # 5. Seleccionar el mejor frame del video para comparar con el carnet
        best_frame = select_best_frame(frames)
        face_comparison = None

        if best_frame is not None:
            # Guardar el mejor frame como imagen temporal
            best_frame_path = os.path.join(TMP_DIR, "best_frame_kyc.jpg")
            cv2.imwrite(best_frame_path, best_frame)

            # 6. Comparar el frame con la imagen del carnet usando el servicio externo
            face_comparison = compare_faces_external(
                source_image_path=carnet_path,
                target_image_path=best_frame_path,
            )
        else:
            liveness["problemas"].append("No se pudo extraer un frame válido para comparación")
            liveness["mensajes"].append("No se encontró un frame con rostro claro para comparar")

        # 7. Determinar resultado final
        liveness_ok  = liveness.get("rostro_detectado", False) and liveness.get("score", 0.0) >= LIVENESS_MIN_SCORE
        face_match_ok = bool(face_comparison and face_comparison.get("verified", False))
        verificado    = liveness_ok and face_match_ok

        # 8. Construir respuesta
        response = {
            "verificado":      verificado,
            "liveness":        liveness,
            "comparacion_rostro": {
                "similarity":  face_comparison.get("similarity")  if face_comparison else None,
                "verified":    face_comparison.get("verified")     if face_comparison else None,
                "source_face": face_comparison.get("source_face") if face_comparison else None,
                "target_face": face_comparison.get("target_face") if face_comparison else None,
                "error":       face_comparison.get("error")       if face_comparison else "No se pudo obtener frame válido",
            },
            "video_convertido": converted,
        }

        if not converted:
            liveness["mensajes"].append(
                "Formato de video no soportado o conversión fallida. Se usó el video original."
            )

        return JSONResponse(content=response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e), "mensajes": ["Ocurrió un error al procesar el KYC"]},
            status_code=500,
        )
    finally:
        # Limpiar archivos temporales
        for path in [carnet_path, video_path, best_frame_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
