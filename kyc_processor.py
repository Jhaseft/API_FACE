import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tempfile
import subprocess
import wave
import webrtcvad
import requests
import mediapipe as mp

# =========================
# Configuración
# =========================
REQUIRE_MIN_FACE_FRAMES_RATIO = 0.40

# Pesos del score de liveness (la similitud viene del servicio externo)
SCORE_W_MOVEMENT = 50.0
SCORE_W_BLINK    = 35.0
SCORE_W_AUDIO    = 15.0

# Umbrales de liveness por movimiento
MOVEMENT_PATH_MAX_PX  = 60.0   # path total para score máximo
MOVEMENT_RANGE_MAX_PX = 20.0   # rango máximo para score máximo
ACTIVE_FRAME_MIN_PX   = 2.0    # desplazamiento mínimo por frame para considerarlo "activo"

# Umbrales de parpadeo
EYE_AR_THRESH           = 0.22
BLINK_MIN_CONSEC_FRAMES = 2
BLINK_MIN_COUNT         = 1

# API externa de comparación de rostros
FACE_COMPARE_API_URL              = "https://servicios-compare-face.b5lsqc.easypanel.host/api/v1/verification/verify"
FACE_COMPARE_API_KEY              = "00000000-0000-0000-0000-000000000004"
FACE_COMPARE_SIMILARITY_THRESHOLD = 0.50   # similarity mínima (0–1) para verificación positiva

# =========================
# MediaPipe
# =========================
mp_fd = mp.solutions.face_detection
mp_fm = mp.solutions.face_mesh

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Índices de landmarks para estimación de pose de cabeza
_NOSE_TIP   = 4
_CHIN       = 152
_L_EYE_OUT  = 33    # esquina exterior ojo izquierdo
_R_EYE_OUT  = 263   # esquina exterior ojo derecho


# =========================
# Conversión de video
# =========================
def convert_video_to_mp4(video_path):
    tmp_mp4 = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                tmp_mp4,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("[FFMPEG ERROR]", result.stderr)
            return video_path
        return tmp_mp4
    except Exception as e:
        print(f"[ERROR] convert_video_to_mp4: {e}")
        return video_path


# =========================
# Audio
# =========================
def check_audio_presence(video_path, aggressiveness=2):
    tmp_audio = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000", "-y", tmp_audio],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False, "Error extrayendo audio"

    if not os.path.exists(tmp_audio):
        return False, "No se creó archivo de audio"

    try:
        wf  = wave.open(tmp_audio, "rb")
        vad = webrtcvad.Vad(aggressiveness)
        frame_duration = 30
        frame_bytes    = int(wf.getframerate() * (frame_duration / 1000.0) * 2)
        audio_data     = wf.readframes(wf.getnframes())
        voiced, total  = 0, 0
        for i in range(0, len(audio_data), frame_bytes):
            frame = audio_data[i : i + frame_bytes]
            if len(frame) < frame_bytes:
                break
            total += 1
            if vad.is_speech(frame, wf.getframerate()):
                voiced += 1
        ratio    = voiced / total if total > 0 else 0.0
        audio_ok = ratio > 0.05
        msg      = (
            f"Voz detectada en {ratio*100:.1f}% del audio"
            if audio_ok
            else "No se detectó voz suficiente"
        )
    except Exception:
        audio_ok, msg = False, "Error leyendo audio"

    return audio_ok, msg


# =========================
# Detección de rostro (centro)
# =========================
def _rect_from_detection(det, w, h, pad=0.20):
    bb = det.location_data.relative_bounding_box
    x1 = max(int((bb.xmin - pad) * w), 0)
    y1 = max(int((bb.ymin - pad) * h), 0)
    x2 = min(int((bb.xmin + bb.width  + pad) * w), w)
    y2 = min(int((bb.ymin + bb.height + pad) * h), h)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _detect_face_center(frame_bgr, detector):
    """Devuelve (cx, cy) del rostro más confiable, o None."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)
    if not res.detections:
        return None
    det = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
    if det.score[0] < 0.7:
        return None
    rect = _rect_from_detection(det, w, h)
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# =========================
# EAR + pose de cabeza
# =========================
def _compute_ear_and_pose(frame_bgr, mesh):
    """
    Retorna (ear, head_angles, has_landmarks).
    head_angles = (yaw_proxy_px, pitch_proxy_px) — desviaciones relativas.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None, None, False

    lm     = res.multi_face_landmarks[0]
    coords = [(p.x * w, p.y * h) for p in lm.landmark]

    def _ear_eye(idx):
        pts = [np.array(coords[i]) for i in idx]
        p1, p2, p3, p4, p5, p6 = pts
        vertical   = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        return vertical / (2.0 * horizontal) if horizontal > 0 else None

    left  = _ear_eye(LEFT_EYE_IDX)
    right = _ear_eye(RIGHT_EYE_IDX)
    ear   = (left + right) / 2.0 if (left is not None and right is not None) else None

    # Pose de cabeza usando nose tip, ojo izq y ojo der
    nose   = np.array(coords[_NOSE_TIP])
    l_eye  = np.array(coords[_L_EYE_OUT])
    r_eye  = np.array(coords[_R_EYE_OUT])
    chin   = np.array(coords[_CHIN])

    eye_center  = (l_eye + r_eye) / 2.0
    eye_span    = float(np.linalg.norm(r_eye - l_eye))
    face_height = float(np.linalg.norm(chin - eye_center))

    # yaw: desplazamiento horizontal de la nariz respecto al centro de los ojos (normalizado por span)
    yaw_proxy   = float((nose[0] - eye_center[0]) / max(eye_span, 1.0))
    # pitch: desplazamiento vertical normalizado por altura del rostro
    pitch_proxy = float((nose[1] - eye_center[1]) / max(face_height, 1.0))

    return ear, (yaw_proxy, pitch_proxy), True


# =========================
# Análisis de liveness por movimiento
# =========================
def _analyze_movement_liveness(centers, head_angles_list):
    """
    Combina trayectoria del centro del rostro y variación de pose para
    estimar liveness por movimiento.
    Retorna (score 0–100, mensaje, detalles).
    """
    details = {
        "total_path_px": 0.0,
        "max_range_px": 0.0,
        "active_frames_ratio": 0.0,
        "head_rotation_score": 0.0,
    }

    if len(centers) < 3:
        return 0.0, "Muy pocos frames con rostro para evaluar movimiento", details

    arr = np.array(centers, dtype=float)

    # — Longitud total de la trayectoria
    diffs         = np.diff(arr, axis=0)
    frame_dists   = np.linalg.norm(diffs, axis=1)
    total_path    = float(np.sum(frame_dists))
    details["total_path_px"] = round(total_path, 2)

    # — Rango máximo (extensión espacial)
    range_x  = float(arr[:, 0].max() - arr[:, 0].min())
    range_y  = float(arr[:, 1].max() - arr[:, 1].min())
    max_range = max(range_x, range_y)
    details["max_range_px"] = round(max_range, 2)

    # — Fracción de frames con desplazamiento notable
    active_frames = float(np.sum(frame_dists > ACTIVE_FRAME_MIN_PX))
    active_ratio  = active_frames / len(frame_dists)
    details["active_frames_ratio"] = round(active_ratio, 3)

    # — Variación de pose de cabeza
    head_rot_score = 0.0
    valid_angles   = [a for a in head_angles_list if a is not None]
    if len(valid_angles) >= 3:
        yaws   = [a[0] for a in valid_angles]
        pitches = [a[1] for a in valid_angles]
        yaw_range   = float(np.max(yaws)   - np.min(yaws))
        pitch_range = float(np.max(pitches) - np.min(pitches))
        # ~0.25 de yaw_range normalizado → plena puntuación de yaw
        # ~0.20 de pitch_range            → plena puntuación de pitch
        head_rot_score = min(
            (yaw_range / 0.25) * 50.0 + (pitch_range / 0.20) * 50.0,
            100.0,
        )
    details["head_rotation_score"] = round(head_rot_score, 2)

    # — Score compuesto
    path_score   = min(total_path  / MOVEMENT_PATH_MAX_PX,  1.0) * 40.0
    range_score  = min(max_range   / MOVEMENT_RANGE_MAX_PX, 1.0) * 30.0
    active_score = min(active_ratio / 0.40,                  1.0) * 15.0
    rot_score    = min(head_rot_score / 100.0,               1.0) * 15.0

    final = round(min(path_score + range_score + active_score + rot_score, 100.0), 2)

    msg = (
        f"Path: {total_path:.1f}px | Rango: {max_range:.1f}px | "
        f"Frames activos: {int(active_frames)}/{len(frame_dists)} | "
        f"Rotación cabeza: {head_rot_score:.1f}"
    )
    return final, msg, details


# =========================
# Comparación externa de rostros
# =========================
def compare_faces_external(source_image_path, target_image_path):
    """
    Llama al servicio externo para comparar rostros.
    source_image_path : imagen del documento/carnet
    target_image_path : selfie o frame del video

    Retorna dict:
      similarity   (float, 0–1)
      verified     (bool)
      source_face  (dict con age/gender/pose/box/landmarks del source, o None)
      target_face  (dict con age/gender/pose/box/landmarks + similarity del mejor match, o None)
      raw_response (respuesta completa de la API)
      error        (str o None)
    """
    result = {
        "similarity":    0.0,
        "verified":      False,
        "source_face":   None,
        "target_face":   None,
        "raw_response":  None,
        "error":         None,
    }
    try:
        params  = {"face_plugins": "landmarks,gender,age,pose"}
        headers = {"x-api-key": FACE_COMPARE_API_KEY}

        def _call_api(src_path, tgt_path):
            with open(src_path, "rb") as src_f, open(tgt_path, "rb") as tgt_f:
                return requests.post(
                    FACE_COMPARE_API_URL,
                    params=params,
                    headers=headers,
                    files={"source_image": src_f, "target_image": tgt_f},
                    timeout=30,
                )

        def _is_multi_face_error(r):
            if r.ok:
                return False
            try:
                return r.json().get("code") == 31
            except Exception:
                return False

        resp = _call_api(source_image_path, target_image_path)

        # Si el source (carnet) tiene múltiples rostros, invertir el orden e intentar de nuevo
        if _is_multi_face_error(resp):
            resp_inv = _call_api(target_image_path, source_image_path)
            if _is_multi_face_error(resp_inv):
                # Ambas combinaciones fallan por múltiples rostros → error definitivo
                result["error"] = "Se detectaron múltiples rostros en el documento. El carnet debe mostrar un único rostro."
                return result
            # La versión invertida funcionó, usarla
            resp = resp_inv

        if not resp.ok:
            resp.raise_for_status()

        data = resp.json()
        result["raw_response"] = data

        api_results = data.get("result", [])
        if not api_results:
            result["error"] = "El servicio no devolvió resultados"
            return result

        first = api_results[0]
        result["source_face"] = first.get("source_image_face")

        matches = first.get("face_matches", [])
        if not matches:
            result["error"] = "No se encontró coincidencia de rostro en la imagen destino"
            return result

        best = max(matches, key=lambda m: m.get("similarity", 0.0))
        similarity           = float(best.get("similarity", 0.0))
        result["similarity"] = round(similarity, 5)
        result["target_face"] = best
        result["verified"]    = similarity >= FACE_COMPARE_SIMILARITY_THRESHOLD

    except requests.exceptions.Timeout:
        result["error"] = "Timeout al conectar con el servicio de comparación de rostros"
    except requests.exceptions.RequestException as e:
        result["error"] = f"Error de red: {e}"
    except Exception as e:
        result["error"] = f"Error inesperado: {e}"

    return result


# =========================
# Selección del mejor frame
# =========================
def select_best_frame(frames):
    """
    Devuelve el frame más adecuado para la comparación externa de rostros:
    - Mayor confianza de detección
    - Desempate por nitidez (varianza del Laplaciano)
    Retorna numpy array BGR, o None si ningún frame tiene rostro válido.
    """
    fd = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.7)
    best_frame, best_score = None, -1.0

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)
        if not res.detections:
            continue
        det = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
        conf = float(det.score[0])
        if conf < 0.7:
            continue
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # 70% confianza + 30% nitidez (normalizada asumiendo max ~1000)
        combined = conf * 0.7 + min(sharpness / 1000.0, 1.0) * 0.3
        if combined > best_score:
            best_score = combined
            best_frame = frame

    fd.close()
    return best_frame


# =========================
# Función principal de liveness
# =========================
def procesar_frames(frames, video_path=None, audio_check=True):
    """
    Analiza liveness en los frames del video:
      - Movimiento del rostro (trayectoria del centro + variación de pose)
      - Parpadeo (Eye Aspect Ratio)
      - Presencia de voz en el audio

    La comparación del rostro con el documento se realiza por separado
    mediante compare_faces_external().

    Retorna dict con liveness_movimiento, parpadeo_detectado, audio, score, etc.
    """
    resultado = {
        "liveness_movimiento": 0.0,
        "parpadeo_detectado":  False,
        "audio":               False,
        "rostro_detectado":    False,
        "score":               0.0,
        "problemas":           [],
        "mensajes":            [],
        "detalles": {
            "frames_totales":    len(frames),
            "frames_con_rostro": 0,
            "face_ratio":        0.0,
            "blink_count":       0,
            "audio_msg":         "",
            "movement":          {},
        },
    }

    if not frames:
        resultado["problemas"].append("No se recibieron frames.")
        return resultado

    # ——— Audio ———————————————————————————————————————
    if audio_check and video_path:
        audio_ok, audio_msg = check_audio_presence(video_path)
    else:
        audio_ok, audio_msg = False, "Audio no verificado"
    resultado["audio"]                    = audio_ok
    resultado["detalles"]["audio_msg"]    = audio_msg
    resultado["mensajes"].append(f"Audio: {audio_msg}")

    # ——— MediaPipe ————————————————————————————————————
    fd = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.7)
    fm = mp_fm.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # ——— Procesar frames ——————————————————————————————
    centers, head_angles_list = [], []
    face_count, blink_count, consec_blink = 0, 0, 0

    for frame in frames:
        center = _detect_face_center(frame, fd)
        if center is not None:
            face_count += 1
            centers.append(center)

        ear, head_angles, has_lm = _compute_ear_and_pose(frame, fm)
        if has_lm:
            if head_angles is not None:
                head_angles_list.append(head_angles)
            if ear is not None:
                if ear < EYE_AR_THRESH:
                    consec_blink += 1
                else:
                    if consec_blink >= BLINK_MIN_CONSEC_FRAMES:
                        blink_count += 1
                    consec_blink = 0

    if consec_blink >= BLINK_MIN_CONSEC_FRAMES:
        blink_count += 1

    total_frames = len(frames)
    face_ratio   = face_count / total_frames if total_frames > 0 else 0.0
    resultado["rostro_detectado"]              = face_ratio >= REQUIRE_MIN_FACE_FRAMES_RATIO and face_count >= 3
    resultado["detalles"]["frames_con_rostro"] = face_count
    resultado["detalles"]["face_ratio"]        = round(face_ratio, 3)
    resultado["detalles"]["blink_count"]       = blink_count

    if resultado["rostro_detectado"]:
        resultado["mensajes"].append(f"Rostro detectado en video ({face_count}/{total_frames} frames)")
    else:
        resultado["mensajes"].append(f"Rostro insuficiente en video ({face_count}/{total_frames})")
        resultado["problemas"].append("Rostro insuficiente en video")

    # ——— Liveness por movimiento ——————————————————————
    liveness_score, liveness_msg, movement_details = _analyze_movement_liveness(
        centers, head_angles_list
    )
    resultado["liveness_movimiento"]      = liveness_score
    resultado["detalles"]["movement"]     = movement_details
    resultado["mensajes"].append(f"Liveness movimiento: {liveness_score:.2f}% — {liveness_msg}")

    if liveness_score < 20.0:
        resultado["problemas"].append("Movimiento insuficiente para confirmar liveness")

    # ——— Parpadeo ————————————————————————————————————
    resultado["parpadeo_detectado"] = blink_count >= BLINK_MIN_COUNT
    if resultado["parpadeo_detectado"]:
        resultado["mensajes"].append(f"Parpadeo detectado: {blink_count} veces")
    else:
        resultado["mensajes"].append("No se detectó parpadeo suficiente")
        resultado["problemas"].append("Sin parpadeo detectado")

    # ——— Score de liveness ————————————————————————————
    if resultado["rostro_detectado"]:
        mov_score   = liveness_score * (SCORE_W_MOVEMENT / 100.0)
        blink_score = SCORE_W_BLINK if resultado["parpadeo_detectado"] else 0.0
        audio_score = SCORE_W_AUDIO if resultado["audio"] else 0.0
        score = mov_score + blink_score + audio_score
    else:
        score = 0.0

    resultado["score"] = round(min(100.0, score), 2)
    return resultado
