import os
import cv2
import numpy as np
import tempfile
import subprocess
import wave
import webrtcvad

# =========================
# Configuración
# =========================
CONFIDENCE_THRESHOLD = 0.5       # Confianza mínima para detección de rostro
REQUIRE_MIN_FACE_FRAMES_RATIO = 0.40  # 40% de frames deben tener rostro
MIN_FACE_FRAMES = 3              # Mínimo de frames con rostro
MOVEMENT_THRESHOLD_PX = 15       # Movimiento mínimo en píxeles
MAX_VERIFY_FRAMES = 5            # Frames a verificar para similitud

# Pesos para scoring
SCORE_W_DETECTION = 40.0         # Detección de rostro
SCORE_W_CONSISTENCY = 30.0       # Consistencia (frames con rostro)
SCORE_W_MOVEMENT = 20.0          # Movimiento detectado
SCORE_W_AUDIO = 10.0             # Audio detectado

# Rutas de modelos (se descargarán automáticamente)
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# URLs de descarga de modelos
MODEL_URLS = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
}

# =========================
# Descargar modelos
# =========================
def download_models():
    """Descarga los modelos de OpenCV DNN si no existen"""
    import urllib.request
    
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            print(f"Descargando {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✓ {filename} descargado")
            except Exception as e:
                print(f"✗ Error descargando {filename}: {e}")
                raise
        else:
            print(f"✓ {filename} ya existe (usando cache)")
    
    return os.path.join(models_dir, FACE_PROTO), os.path.join(models_dir, FACE_MODEL)

# Inicializar modelo al cargar el módulo
try:
    PROTO_PATH, MODEL_PATH = download_models()
    FACE_NET = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
    print("✓ Modelo de detección facial cargado correctamente")
except Exception as e:
    print(f"✗ Error cargando modelo: {e}")
    FACE_NET = None

# =========================
# Audio
# =========================
def check_audio_presence(video_path, aggressiveness=2):
    """Verifica si hay audio/voz en el video"""
    tmp_audio = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
    ffmpeg_bin = "ffmpeg"
    
    try:
        subprocess.run([
            ffmpeg_bin, "-i", video_path,
            "-ac", "1", "-ar", "16000",
            "-y", tmp_audio
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        return False, "Error extrayendo audio"
    
    if not os.path.exists(tmp_audio):
        return False, "No se creó archivo de audio"
    
    try:
        wf = wave.open(tmp_audio, 'rb')
        vad = webrtcvad.Vad(aggressiveness)
        frame_duration = 30
        frame_bytes = int(wf.getframerate() * (frame_duration / 1000.0) * 2)
        audio_data = wf.readframes(wf.getnframes())
        
        voiced_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_data), frame_bytes):
            frame = audio_data[i:i+frame_bytes]
            if len(frame) < frame_bytes:
                break
            total_frames += 1
            if vad.is_speech(frame, wf.getframerate()):
                voiced_frames += 1
        
        voice_ratio = voiced_frames / total_frames if total_frames > 0 else 0.0
        audio_ok = voice_ratio > 0.05
        audio_msg = f"Voz detectada en {voice_ratio*100:.1f}% del audio" if audio_ok else "No se detectó voz suficiente"
        
        wf.close()
        os.remove(tmp_audio)
    except Exception as e:
        audio_ok = False
        audio_msg = f"Error leyendo audio: {e}"
    
    return audio_ok, audio_msg

# =========================
# Detección de rostros
# =========================
def detect_faces(frame, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Detecta rostros en un frame usando OpenCV DNN
    
    Returns:
        list: Lista de detecciones, cada una con formato:
              {'bbox': (x, y, w, h), 'confidence': float, 'center': (cx, cy)}
    """
    if FACE_NET is None:
        return []
    
    h, w = frame.shape[:2]
    
    # Preparar imagen para el modelo
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 
        1.0, 
        (300, 300), 
        (104.0, 177.0, 123.0)
    )
    
    FACE_NET.setInput(blob)
    detections = FACE_NET.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            # Asegurar que las coordenadas estén dentro de los límites
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width / 2
                center_y = y1 + height / 2
                
                faces.append({
                    'bbox': (x1, y1, width, height),
                    'confidence': float(confidence),
                    'center': (center_x, center_y)
                })
    
    return faces

def crop_face(frame, bbox, padding=0.2):
    """
    Recorta el rostro de un frame con padding
    
    Args:
        frame: Frame BGR
        bbox: (x, y, w, h)
        padding: Porcentaje de padding adicional
    
    Returns:
        np.array: Imagen recortada del rostro
    """
    x, y, w, h = bbox
    h_frame, w_frame = frame.shape[:2]
    
    # Calcular padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    # Aplicar padding
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_frame, x + w + pad_w)
    y2 = min(h_frame, y + h + pad_h)
    
    return frame[y1:y2, x1:x2]

def compute_face_similarity(face1, face2):
    """
    Calcula similitud entre dos rostros usando histogramas de color
    
    Args:
        face1, face2: Imágenes BGR de rostros
    
    Returns:
        float: Similitud en porcentaje (0-100)
    """
    if face1 is None or face2 is None or face1.size == 0 or face2.size == 0:
        return 0.0
    
    try:
        # Redimensionar a tamaño estándar
        size = (128, 128)
        face1_resized = cv2.resize(face1, size)
        face2_resized = cv2.resize(face2, size)
        
        # Convertir a HSV para mejor comparación
        face1_hsv = cv2.cvtColor(face1_resized, cv2.COLOR_BGR2HSV)
        face2_hsv = cv2.cvtColor(face2_resized, cv2.COLOR_BGR2HSV)
        
        # Calcular histogramas
        hist1 = cv2.calcHist([face1_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([face2_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
        # Normalizar
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Comparar usando correlación
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Convertir a porcentaje
        return max(0.0, min(100.0, similarity * 100.0))
    
    except Exception as e:
        print(f"Error calculando similitud: {e}")
        return 0.0

# =========================
# Función principal
# =========================
def procesar_frames(frames, carnet_frente_path, video_path=None, audio_check=True):
    """
    Procesa frames del video y la imagen del carnet para verificación KYC
    
    Args:
        frames: Lista de frames BGR del video
        carnet_frente_path: Path a la imagen del carnet
        video_path: Path al video (para análisis de audio)
        audio_check: Si debe verificar audio
    
    Returns:
        dict: Resultado de la verificación con score y detalles
    """
    resultado = {
        "verificado": False,
        "similitud_promedio": 0.0,
        "liveness_movimiento": 0.0,
        "rostro_detectado": False,
        "audio": False,
        "score": 0.0,
        "problemas": [],
        "mensajes": [],
        "detalles": {
            "frames_totales": len(frames),
            "frames_con_rostro": 0,
            "num_rostros_promedio": 0.0,
            "face_ratio": 0.0,
            "audio_msg": ""
        }
    }
    
    if not frames:
        resultado["problemas"].append("No se recibieron frames.")
        return resultado
    
    # ===================== Audio =====================
    if audio_check and video_path:
        audio_ok, audio_msg = check_audio_presence(video_path)
    else:
        audio_ok, audio_msg = False, "Audio no verificado"
    
    resultado["audio"] = audio_ok
    resultado["detalles"]["audio_msg"] = audio_msg
    resultado["mensajes"].append(f"Audio: {audio_msg}")
    
    # ===================== Procesar imagen del carnet =====================
    carnet_img = cv2.imread(carnet_frente_path)
    if carnet_img is None:
        resultado["problemas"].append("No se pudo leer la imagen del carnet.")
        resultado["mensajes"].append("❌ No se pudo leer la imagen del carnet")
        return resultado
    
    carnet_faces = detect_faces(carnet_img, confidence_threshold=0.3)  # Más permisivo para carnet
    
    if carnet_faces:
        # Tomar el rostro con mayor confianza
        carnet_face = max(carnet_faces, key=lambda f: f['confidence'])
        carnet_crop = crop_face(carnet_img, carnet_face['bbox'])
        resultado["mensajes"].append(f"✓ Rostro detectado en carnet (confianza: {carnet_face['confidence']:.2%})")
    else:
        resultado["mensajes"].append("⚠️ No se detectó rostro en imagen del carnet")
        resultado["problemas"].append("No hay rostro detectable en la imagen del carnet")
        carnet_crop = None
    
    # ===================== Procesar frames del video =====================
    face_frames = []
    centers = []
    face_count = 0
    total_faces_detected = 0
    
    for frame in frames:
        faces = detect_faces(frame)
        
        if faces:
            face_count += 1
            total_faces_detected += len(faces)
            
            # Tomar el rostro con mayor confianza
            best_face = max(faces, key=lambda f: f['confidence'])
            centers.append(best_face['center'])
            
            face_crop = crop_face(frame, best_face['bbox'])
            face_frames.append({
                'crop': face_crop,
                'center': best_face['center'],
                'confidence': best_face['confidence']
            })
    
    total_frames = len(frames)
    face_ratio = face_count / total_frames if total_frames > 0 else 0.0
    avg_faces = total_faces_detected / total_frames if total_frames > 0 else 0.0
    
    resultado["rostro_detectado"] = (face_ratio >= REQUIRE_MIN_FACE_FRAMES_RATIO and face_count >= MIN_FACE_FRAMES)
    resultado["detalles"]["frames_con_rostro"] = face_count
    resultado["detalles"]["num_rostros_promedio"] = round(avg_faces, 2)
    resultado["detalles"]["face_ratio"] = round(face_ratio, 3)
    
    if resultado["rostro_detectado"]:
        resultado["mensajes"].append(f"✓ Rostro detectado en video ({face_count}/{total_frames} frames = {face_ratio:.1%})")
    else:
        resultado["mensajes"].append(f"⚠️ Rostro insuficiente en video ({face_count}/{total_frames} = {face_ratio:.1%})")
        resultado["problemas"].append("Rostro insuficiente en video")
    
    # ===================== Similitud =====================
    similitudes = []
    if carnet_crop is not None and face_frames:
        # Seleccionar frames distribuidos uniformemente
        num_frames_to_compare = min(MAX_VERIFY_FRAMES, len(face_frames))
        indices = np.linspace(0, len(face_frames) - 1, num=num_frames_to_compare, dtype=int)
        
        for idx in indices:
            face_frame = face_frames[idx]
            sim = compute_face_similarity(face_frame['crop'], carnet_crop)
            if sim is not None:
                similitudes.append(sim)
        
        if similitudes:
            resultado["similitud_promedio"] = float(np.mean(similitudes))
            resultado["mensajes"].append(f"✓ Similitud promedio: {resultado['similitud_promedio']:.2f}%")
        else:
            resultado["mensajes"].append("⚠️ No se pudo calcular similitud")
            resultado["problemas"].append("Similitud no calculada")
    else:
        resultado["mensajes"].append("⚠️ No se puede calcular similitud")
        resultado["problemas"].append("Similitud no calculada")
    
    # ===================== Liveness (movimiento) =====================
    liveness_score = 0.0
    if len(centers) >= 5:
        # Calcular movimiento usando últimos 5 frames
        recent_centers = centers[-5:]
        delta_x = max([c[0] for c in recent_centers]) - min([c[0] for c in recent_centers])
        delta_y = max([c[1] for c in recent_centers]) - min([c[1] for c in recent_centers])
        delta = max(delta_x, delta_y)
        
        liveness_score = min(delta / MOVEMENT_THRESHOLD_PX * 100.0, 100.0)
        resultado["liveness_movimiento"] = round(liveness_score, 2)
        resultado["mensajes"].append(f"✓ Movimiento detectado: {resultado['liveness_movimiento']:.2f}%")
    elif resultado["rostro_detectado"]:
        resultado["problemas"].append("Movimiento insuficiente")
        resultado["mensajes"].append("⚠️ Movimiento insuficiente para evaluar liveness")
    
    # ===================== Cálculo de Score =====================
    score = 0.0
    
    if carnet_crop is not None and resultado["rostro_detectado"]:
        # Score por detección (si hay rostro en video)
        detection_score = SCORE_W_DETECTION
        
        # Score por consistencia (proporción de frames con rostro)
        consistency_score = face_ratio * SCORE_W_CONSISTENCY
        
        # Score por movimiento
        movement_score = min(liveness_score / 100.0, 1.0) * SCORE_W_MOVEMENT
        
        # Score por audio
        audio_score = SCORE_W_AUDIO if resultado["audio"] else 0.0
        
        score = detection_score + consistency_score + movement_score + audio_score
        
        # Criterios de verificación
        verificado = (
            resultado["rostro_detectado"] and
            face_ratio >= REQUIRE_MIN_FACE_FRAMES_RATIO and
            (liveness_score >= 20.0 or resultado["audio"])
        )
        
        resultado["verificado"] = verificado
    
    resultado["score"] = round(min(100.0, score), 2)
    
    return resultado