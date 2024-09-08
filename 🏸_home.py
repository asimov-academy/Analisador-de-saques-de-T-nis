import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Asimov Serve Analyzer! 🎾")

st.markdown(
    """
    Bem vindo ao Asimov Serve Analyzer! Este é um aplicativo que detecta automaticamente
    saques em uma sessão de treino e permite a comparação dos movimentos com um
    vídeo de referência.

    ### Webcam Live Feeder
    Utilize sua Webcam para gravar suas seções de treino. 
    Seus saques serão automaticamente cortados e organizados por data.
"""
)




import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import os
import csv
import pickle
from collections import deque


mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

run = st.checkbox('Run')
col1, col2 = st.columns(2)
sess_date = str(col1.date_input("Session Date"))
cam = col2.selectbox("Camera", ('Webcam', 'Iphone'))
cam_dict = {"Webcam": 1, "Iphone": 0}

FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(cam_dict[cam])

# Directories
serve_dir = os.path.join("sessions", "serve_" + sess_date)
land_dir = os.path.join("sessions", "land_" + sess_date)
os.makedirs(serve_dir, exist_ok=True)
os.makedirs(land_dir, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
min_visibility = 0.9

frames = deque(maxlen=30*4)
landmarks_frames = deque(maxlen=30*4)

condition1_met = condition2_met = condition3_met = False
saque_count = [int(i.split("_")[1].split(".")[0]) for i in os.listdir("sessions/serve_2023-11-02")][-1]

os.listdir("sessions")
post_condition3_frames = 0  # Novo contador para 2 segundos extras

while run:
    _, frame = cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    results = pose.process(frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        landmarks_frames.append(results.pose_landmarks)
        frames.append(frame)

        # Verificações das condições com visibilidade mínima
        wrist_visible = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].visibility > min_visibility
        elbow_visible = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility > min_visibility
        shoulder_visible = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > min_visibility

        # Condição 1
        if wrist_visible and landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y < landmarks[mp_pose.PoseLandmark.NOSE].y:
            condition1_met = True

        # Condição 2
        if condition1_met and elbow_visible and shoulder_visible and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y:
            condition2_met = True

        # Condição 3
        if condition2_met and elbow_visible and shoulder_visible and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y:
            condition3_met = True

        # Se a condição 3 foi atendida, começar a contar os frames para os 2 segundos extras
        if condition3_met:
            post_condition3_frames += 1

        # Se todas as condições foram atendidas, começa a gravar
        if condition1_met and condition2_met and condition3_met and out is None:
            h, w, _ = frame.shape
            saque_count += 1
            out = cv2.VideoWriter(f'{serve_dir}/saque_{saque_count}.mp4', fourcc, fps, (w, h))

        # Se a gravação começou, escreva os frames no arquivo
        if out is not None:            
            # Continue gravando por 2 segundos após a condição 3 ser verificada
            if post_condition3_frames >= fps * 2:
                with open(f'{land_dir}/landmarks_saque_{saque_count}.pickle', 'wb') as f:
                    pickle.dump(list(landmarks_frames), f)

                while not len(frames) == 0:
                    out.write(frames.popleft())

                out.release()
                out = None
                print(f"Saque {saque_count} salvo")
                # Resetando as condições e contadores para o próximo saque
                condition1_met = condition2_met = condition3_met = False
                post_condition3_frames = 0
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    
else:
    st.write('Stopped')