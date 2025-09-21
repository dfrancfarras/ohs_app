import cv2
import numpy as np
import pandas as pd
import math
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
def angle_abc(a, b, c):
    if a is None or b is None or c is None:
        return np.nan
    ba = a - b
    bc = c - b
    ang = math.degrees(math.atan2(bc[1], bc[0]) - math.atan2(ba[1], ba[0]))
    ang = abs(ang)
    return 360 - ang if ang > 180 else ang

def angle_with_vertical(p1, p2):
    if p1 is None or p2 is None:
        return np.nan
    v = p2 - p1
    if np.linalg.norm(v) < 1e-6:
        return np.nan
    v = v / (np.linalg.norm(v) + 1e-9)
    vert = np.array([0, -1.0])
    cosang = np.clip(np.dot(v, vert), -1.0, 1.0)
    return math.degrees(math.acos(cosang))
LM = mp_pose.PoseLandmark

def get_xy_vis(landmarks, idx, w, h):
    l = landmarks[idx.value]
    return np.array([l.x * w, l.y * h]), float(l.visibility)

def weighted_avg(val_left, w_left, val_right, w_right):
    w_sum = w_left + w_right
    if w_sum < 1e-6:
        return np.nan
    return (val_left * w_left + val_right * w_right) / w_sum

def compute_ohs_angles(landmarks, w, h):
    # Lado izquierdo
    hip_L, vis_hip_L = get_xy_vis(landmarks, LM.LEFT_HIP, w, h)
    knee_L, vis_knee_L = get_xy_vis(landmarks, LM.LEFT_KNEE, w, h)
    ankle_L, vis_ank_L = get_xy_vis(landmarks, LM.LEFT_ANKLE, w, h)
    foot_L, vis_foot_L = get_xy_vis(landmarks, LM.LEFT_FOOT_INDEX, w, h)
    shoulder_L, vis_shL = get_xy_vis(landmarks, LM.LEFT_SHOULDER, w, h)
    elbow_L, vis_elb_L = get_xy_vis(landmarks, LM.LEFT_ELBOW, w, h)

    # Lado derecho
    hip_R, vis_hip_R = get_xy_vis(landmarks, LM.RIGHT_HIP, w, h)
    knee_R, vis_knee_R = get_xy_vis(landmarks, LM.RIGHT_KNEE, w, h)
    ankle_R, vis_ank_R = get_xy_vis(landmarks, LM.RIGHT_ANKLE, w, h)
    foot_R, vis_foot_R = get_xy_vis(landmarks, LM.RIGHT_FOOT_INDEX, w, h)
    shoulder_R, vis_shR = get_xy_vis(landmarks, LM.RIGHT_SHOULDER, w, h)
    elbow_R, vis_elb_R = get_xy_vis(landmarks, LM.RIGHT_ELBOW, w, h)

    # Ángulos
    knee_L_ang = angle_abc(hip_L, knee_L, ankle_L)
    knee_R_ang = angle_abc(hip_R, knee_R, ankle_R)
    hip_L_ang = angle_abc(shoulder_L, hip_L, knee_L)
    hip_R_ang = angle_abc(shoulder_R, hip_R, knee_R)
    ankle_L_ang = 180 - angle_abc(knee_L, ankle_L, foot_L)
    ankle_R_ang = 180 - angle_abc(knee_R, ankle_R, foot_R)
    shoulder_L_ang = angle_abc(hip_L, shoulder_L, elbow_L)
    shoulder_R_ang = angle_abc(hip_R, shoulder_R, elbow_R)
    trunk_L_ang = angle_with_vertical(hip_L, shoulder_L)
    trunk_R_ang = angle_with_vertical(hip_R, shoulder_R)

    return {
        "knee_flexion": weighted_avg(knee_L_ang, vis_knee_L, knee_R_ang, vis_knee_R),
        "hip_flexion": weighted_avg(hip_L_ang, vis_hip_L, hip_R_ang, vis_hip_R),
        "ankle_dorsiflex": weighted_avg(ankle_L_ang, vis_ank_L, ankle_R_ang, vis_ank_R),
        "shoulder_elev": weighted_avg(shoulder_L_ang, vis_shL, shoulder_R_ang, vis_shR),
        "trunk_lean": weighted_avg(trunk_L_ang, vis_hip_L, trunk_R_ang, vis_hip_R),
    }
def extract_angles_from_video(video_path, show_preview=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("No se pudo leer el primer frame.")
    h0, w0 = frame0.shape[:2]

    results_rows = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            row = {"frame": frame_idx, "time_s": frame_idx/fps}
            if res.pose_landmarks:
                angles = compute_ohs_angles(res.pose_landmarks.landmark, w, h)
                row.update(angles)
            results_rows.append(row)
            frame_idx += 1

    cap.release()
    return pd.DataFrame(results_rows)

def get_deepest_ohs_angles(df):
    idx_min = df["knee_flexion"].idxmin()
    return df.loc[[idx_min]]
def save_deepest_frame_with_angles(video_path, df, model, output_img="ohs_deepest_result.png"):
    idx_min = df["knee_flexion"].idxmin()
    frame_target = int(df.loc[idx_min, "frame"])

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_target)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("No se pudo leer el frame objetivo.")

    h, w = frame.shape[:2]
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
            )

    # Preparar valores
    row = df.loc[idx_min]
    X = pd.DataFrame([{
        "ankle_dorsiflex": row["ankle_dorsiflex"],
        "knee_flexion": row["knee_flexion"],
        "hip_flexion": row["hip_flexion"],
        "trunk_lean": row["trunk_lean"],
        "shoulder_elev": row["shoulder_elev"]
    }])

    pred = model.predict(X)[0]

    # Color clasificación
    color = (0,255,0) if pred == "Excelente" else (0,255,255) if pred == "Normal" else (0,0,255)
    cv2.putText(frame, f"Clasificación: {pred}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Guardar imagen
    cv2.imwrite(output_img, frame)

    # Crear tabla resultado
    tabla = pd.DataFrame([{
        "Articulación": "Rodilla", "Valor (deg)": row["knee_flexion"], "Interpretación": "-"
    },{
        "Articulación": "Cadera", "Valor (deg)": row["hip_flexion"], "Interpretación": "-"
    },{
        "Articulación": "Tobillo", "Valor (deg)": row["ankle_dorsiflex"], "Interpretación": "-"
    },{
        "Articulación": "Tronco", "Valor (deg)": row["trunk_lean"], "Interpretación": "-"
    },{
        "Articulación": "Hombro", "Valor (deg)": row["shoulder_elev"], "Interpretación": "-"
    }])

    return tabla
