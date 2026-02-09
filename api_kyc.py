from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
import subprocess
import cv2
from kyc_processor import procesar_frames

app = FastAPI(
    title="KYC Processor API",
    description="API de verificación KYC sin dependencias AVX",
    version="2.0.0"
)

# =========================
# CORS
# =========================
origins = [
    "https://transfers.elchangarrodelima.com",
    "http://localhost:3000",
    "http://localhost:8000"
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
    mp4_path = os.path.join(TMP_DIR, "temp_video_converted.mp4")
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
            text=True,
            timeout=30  # Timeout de 30 segundos
        )
        
        if result.returncode != 0:
            print("[FFMPEG ERROR]", result.stderr)
            return video_path, False
        
        if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
            return mp4_path, True
        else:
            return video_path, False
    
    except subprocess.TimeoutExpired:
        print("[ERROR] Timeout al convertir video")
        return video_path, False
    except Exception as e:
        print(f"[ERROR] convert_video_to_mp4: {e}")
        return video_path, False

def extract_frames_from_video(video_path: str, max_frames: int = 30, frame_skip: int = 5):
    """
    Extrae frames del video con muestreo uniforme.
    
    Args:
        video_path: Path al video
        max_frames: Máximo número de frames a extraer
        frame_skip: Saltar cada N frames
    
    Returns:
        list: Lista de frames BGR
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video para extracción de frames")
    
    # Obtener información del video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {total_frames} frames totales, {fps} FPS")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Muestrear frames uniformemente
        if frame_count % frame_skip == 0 and len(frames) < max_frames:
            # Redimensionar para acelerar procesamiento (mantiene proporción)
            h, w = frame.shape[:2]
            max_dimension = 640
            
            if max(h, w) > max_dimension:
                if w > h:
                    new_w = max_dimension
                    new_h = int(h * (max_dimension / w))
                else:
                    new_h = max_dimension
                    new_w = int(w * (max_dimension / h))
                
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    
    if not frames:
        raise RuntimeError("No se pudieron extraer frames del video")
    
    print(f"Extraídos {len(frames)} frames del video")
    return frames

# =========================
# Endpoints
# =========================
@app.get("/")
async def root():
    return {
        "message": "KYC Processor API v2.0",
        "status": "running",
        "features": [
            "Detección facial OpenCV DNN (sin AVX)",
            "Verificación de liveness por movimiento",
            "Análisis de audio",
            "Scoring automático"
        ]
    }

@app.get("/health")
async def health_check():
    """Endpoint de health check"""
    return {
        "status": "healthy",
        "version": "2.0.0"
    }

@app.post("/registro-face/verify")
async def verify_kyc(
    carnet: UploadFile = File(..., description="Imagen del carnet (frente)"),
    video: UploadFile = File(..., description="Video del rostro del usuario"),
):
    """
    Verifica la identidad del usuario comparando el carnet con el video.
    
    Retorna:
        - verificado: bool - Si pasó la verificación
        - score: float - Puntuación de 0-100
        - similitud_promedio: float - Similitud entre rostros (0-100)
        - liveness_movimiento: float - Score de movimiento detectado
        - rostro_detectado: bool - Si se detectó rostro en el video
        - audio: bool - Si se detectó audio/voz
        - mensajes: list - Mensajes informativos
        - problemas: list - Problemas detectados
        - detalles: dict - Información detallada del procesamiento
    """
    carnet_path = None
    video_path = None
    video_mp4_path = None
    
    try:
        # Guardar archivos temporales
        carnet_path = save_upload_file(carnet)
        video_path = save_upload_file(video)
        
        print(f"📁 Carnet guardado en: {carnet_path}")
        print(f"📁 Video guardado en: {video_path}")
        
        # Verificar que los archivos existen
        if not os.path.exists(carnet_path):
            raise RuntimeError("No se pudo guardar la imagen del carnet")
        if not os.path.exists(video_path):
            raise RuntimeError("No se pudo guardar el video")
        
        # Convertir video a MP4 si es necesario
        video_mp4_path, converted = convert_video_to_mp4(video_path)
        
        if converted:
            print("✓ Video convertido a MP4")
        else:
            print("⚠️ Usando video original (no se pudo convertir)")
        
        # Extraer frames
        print("🎬 Extrayendo frames del video...")
        frames = extract_frames_from_video(video_mp4_path)
        print(f"✓ {len(frames)} frames extraídos")
        
        # Procesar KYC
        print("🔍 Procesando verificación KYC...")
        resultado = procesar_frames(
            frames, 
            carnet_path, 
            video_path=video_mp4_path,
            audio_check=True
        )
        
        # Información adicional
        resultado["video_convertido"] = converted
        resultado["api_version"] = "2.0.0"
        
        if not converted:
            resultado["mensajes"].append(
                "⚠️ Video usado en formato original (conversión no exitosa)"
            )
        
        # Resumen
        print(f"{'✅' if resultado['verificado'] else '❌'} Verificado: {resultado['verificado']}")
        print(f"📊 Score: {resultado['score']}/100")
        print(f"👤 Rostro detectado: {resultado['rostro_detectado']}")
        print(f"🔊 Audio: {resultado['audio']}")
        
        return JSONResponse(content=resultado)
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ ERROR: {error_detail}")
        
        return JSONResponse(
            content={
                "error": str(e),
                "error_type": type(e).__name__,
                "mensajes": ["Ocurrió un error al procesar el KYC"],
                "verificado": False,
                "score": 0.0
            },
            status_code=500
        )
    
    finally:
        # Limpiar archivos temporales
        for path in [carnet_path, video_path, video_mp4_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

@app.post("/registro-face/detect")
async def detect_face_only(
    image: UploadFile = File(..., description="Imagen para detectar rostros")
):
    """
    Endpoint simple para solo detectar rostros en una imagen.
    
    Retorna:
        - rostros_detectados: int - Número de rostros
        - rostros: list - Lista de rostros con bbox y confianza
    """
    image_path = None
    
    try:
        from kyc_processor import detect_faces
        
        # Guardar imagen temporal
        image_path = save_upload_file(image)
        
        # Leer imagen
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("No se pudo leer la imagen")
        
        # Detectar rostros
        faces = detect_faces(img)
        
        resultado = {
            "rostros_detectados": len(faces),
            "rostros": [
                {
                    "bbox": {
                        "x": int(f['bbox'][0]),
                        "y": int(f['bbox'][1]),
                        "width": int(f['bbox'][2]),
                        "height": int(f['bbox'][3])
                    },
                    "confidence": round(f['confidence'], 4),
                    "center": {
                        "x": round(f['center'][0], 2),
                        "y": round(f['center'][1], 2)
                    }
                }
                for f in faces
            ]
        }
        
        return JSONResponse(content=resultado)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        
        return JSONResponse(
            content={
                "error": str(e),
                "rostros_detectados": 0,
                "rostros": []
            },
            status_code=500
        )
    
    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)