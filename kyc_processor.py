# kyc_processor_light.py
# ---------------------------------------------------------
# Procesamiento de frames para KYC (sin transcripción)
# Autor: Jhaseft Saat
# ---------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # no usar GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tempfile
import subprocess
import wave
import webrtcvad
from deepface import DeepFace
import mediapipe as mp

# =========================
# Configuración y umbrales
# =========================
SIMILARITY_REQUIRED = 65.0
REQUIRE_MIN_FACE_FRAMES_RATIO = 0.30
MAX_VERIFY_FRAMES = 5

SCORE_W_SIMILARITY = 40.0
SCORE_W_MOVEMENT   = 30.0
SCORE_W_BLINK      = 20.0
SCORE_W_AUDIO      = 10.0

MOVEMENT_THRESHOLD_PX = 15
EYE_AR_THRESH = 0.22
BLINK_MIN_CONSEC_FRAMES = 2
BLINK_MIN_COUNT = 1

DEEPFACE_MODEL = "Facenet"
DEEPFACE_ENFORCE_DET = False

# =========================
# MediaPipe
# =========================
mp_fd = mp.solutions.face_detection
mp_fm = mp.solutions.face_mesh

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# =========================
# Funciones utilitarias
# =========================
def convert_video_to_mp4(input_path):
    base, _ = os.path.splitext(input_path)
    output_path = base + "_converted.mp4"
    ffmpeg_bin = "ffmpeg"
    subprocess.run([
        ffmpeg_bin,
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        "-y", output_path
    ], check=True)
    return output_path

def check_audio_presence(video_path, aggressiveness=2):
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
    except:
        audio_ok = False
        audio_msg = "Error leyendo audio"

    return audio_ok, audio_msg

# Funciones de rostro
def _rect_from_detection(det, w, h, pad=0.20):
    bb = det.location_data.relative_bounding_box
    x1 = max(int((bb.xmin - pad) * w), 0)
    y1 = max(int((bb.ymin - pad) * h), 0)
    x2 = min(int((bb.xmin + bb.width + pad) * w), w)
    y2 = min(int((bb.ymin + bb.height + pad) * h), h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def _crop_face_with_mediapipe_bgr(frame_bgr, detector):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)
    if not res.detections:
        return None, None
    det = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
    rect = _rect_from_detection(det, w, h, pad=0.20)
    if rect is None:
        return None, None
    x1, y1, x2, y2 = rect
    crop = frame_bgr[y1:y2, x1:x2]
    cx = (x1 + x2) / 2.0
    return crop, cx

def _compute_ear_from_mesh(frame_bgr, mesh):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None, False
    lm = res.multi_face_landmarks[0]
    coords = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]

    def _ear(eye_idx):
        p1, p2, p3, p4, p5, p6 = [coords[i] for i in eye_idx]
        vertical = np.linalg.norm(np.array(p2) - np.array(p6)) + np.linalg.norm(np.array(p3) - np.array(p5))
        horizontal = np.linalg.norm(np.array(p1) - np.array(p4))
        if horizontal == 0: return None
        return vertical / (2.0 * horizontal)

    left = _ear(LEFT_EYE_IDX)
    right = _ear(RIGHT_EYE_IDX)
    if left is None or right is None: return None, False
    return (left + right)/2.0, True

def _resize_for_deepface(img_bgr, size=224):
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)

def _compute_similarity_percent(img1_bgr, img2_bgr):
    try:
        res = DeepFace.verify(img1_bgr, img2_bgr, model_name=DEEPFACE_MODEL, enforce_detection=DEEPFACE_ENFORCE_DET)
        sim = (1.0 - float(res["distance"])) * 100.0
        return float(max(0.0, min(100.0, sim)))
    except:
        return None

# =========================
# Función principal
# =========================
def procesar_frames(frames, carnet_frente_path, video_path=None, audio_check=True):
    resultado = {
        "verificado": False,
        "similitud_promedio": 0.0,
        "liveness_movimiento": 0.0,
        "parpadeo_detectado": False,
        "audio": False,
        "rostro_detectado": False,
        "score": 0.0,
        "problemas": [],
        "detalles": {"frames_totales": len(frames), "frames_con_rostro":0, "face_ratio":0.0, "blink_count":0, "audio_msg":""},
        "mensajes": []
    }

    if not frames:
        resultado["problemas"].append("No se recibieron frames.")
        return resultado

    # Audio
    if audio_check and video_path:
        audio_ok, audio_msg = check_audio_presence(video_path)
    else:
        audio_ok, audio_msg = False, "Audio no verificado"
    resultado["audio"] = audio_ok
    resultado["detalles"]["audio_msg"] = audio_msg
    if not audio_ok:
        resultado["problemas"].append(audio_msg)

    # MediaPipe
    fd = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    fm = mp_fm.FaceMesh(max_num_faces=1, refine_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Preparar carnet
    carnet_img = cv2.imread(carnet_frente_path)
    if carnet_img is None:
        resultado["problemas"].append("No se pudo leer la imagen del carnet.")
        return resultado
    carnet_crop, _ = _crop_face_with_mediapipe_bgr(carnet_img, fd)
    carnet_crop = _resize_for_deepface(carnet_crop if carnet_crop is not None else carnet_img)

    # Procesar frames
    face_frames, centers_x = [], []
    face_count, blink_count, consec_blink = 0, 0, 0
    for idx, f in enumerate(frames):
        if idx % 2 != 0: continue
        face_crop, cx = _crop_face_with_mediapipe_bgr(f, fd)
        if face_crop is not None:
            face_count += 1
            centers_x.append(cx)
            face_frames.append((_resize_for_deepface(face_crop), cx))

        ear, has_landmarks = _compute_ear_from_mesh(f, fm)
        if has_landmarks and ear is not None:
            if ear < EYE_AR_THRESH:
                consec_blink += 1
            else:
                if consec_blink >= BLINK_MIN_CONSEC_FRAMES: blink_count += 1
                consec_blink = 0
    if consec_blink >= BLINK_MIN_CONSEC_FRAMES: blink_count +=1

    total_frames = len(frames)
    face_ratio = face_count/total_frames if total_frames>0 else 0.0
    resultado["rostro_detectado"] = face_ratio >= REQUIRE_MIN_FACE_FRAMES_RATIO
    resultado["detalles"]["frames_con_rostro"] = face_count
    resultado["detalles"]["face_ratio"] = round(face_ratio,3)
    resultado["detalles"]["blink_count"] = blink_count
    if not resultado["rostro_detectado"]:
        resultado["problemas"].append(f"No se detectó rostro suficiente ({face_count}/{total_frames}).")

    # Similitud
    similitudes = []
    if face_frames:
        idxs = np.linspace(0,len(face_frames)-1,num=min(MAX_VERIFY_FRAMES,len(face_frames)),dtype=int)
        for i in idxs:
            sim = _compute_similarity_percent(face_frames[i][0], carnet_crop)
            if sim is not None: similitudes.append(sim)
    resultado["similitud_promedio"] = float(np.mean(similitudes)) if similitudes else 0.0
    if not similitudes: resultado["problemas"].append("No se pudo calcular similitud.")

    # Movimiento
    if len(centers_x) >= 5:
        delta = float(max(centers_x[-5:]) - min(centers_x[-5:]))
        resultado["liveness_movimiento"] = float(min(delta / MOVEMENT_THRESHOLD_PX * 100.0, 100.0))
    elif resultado["rostro_detectado"]:
        resultado["problemas"].append("Movimiento insuficiente para evaluar liveness.")

    resultado["parpadeo_detectado"] = blink_count >= BLINK_MIN_COUNT

    # Score
    score = 0.0
    if resultado["similitud_promedio"]>=SIMILARITY_REQUIRED and resultado["rostro_detectado"] and audio_ok:
        score += min(resultado["similitud_promedio"],100.0)*(SCORE_W_SIMILARITY/100.0)
        score += min(resultado["liveness_movimiento"],100.0)*(SCORE_W_MOVEMENT/100.0)
        if resultado["parpadeo_detectado"]: score+=SCORE_W_BLINK
        if resultado["audio"]: score+=SCORE_W_AUDIO
        resultado["verificado"] = True
    resultado["score"] = round(min(100.0,score),2)
    resultado["similitud_promedio"] = round(resultado["similitud_promedio"],2)
    resultado["liveness_movimiento"] = round(resultado["liveness_movimiento"],2)

    return resultado
