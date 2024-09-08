import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import pickle
from collections import deque


mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
serve_dir = 'serve'
land_dir = 'landmarks'

# Criar uma pasta para salvar os vídeos dos saques se não existir
filename = '04-01_1.MOV'
# filename = 'kyrgios.mp4'

cap = cv2.VideoCapture(filename)

os.makedirs(serve_dir, exist_ok=True)
os.makedirs(land_dir, exist_ok=True)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
min_visibility = 0.9

# Obtém o número total de quadros
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_to_start = int(total_frames * 0.01)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_start)

frames = deque(maxlen=fps*4)
landmarks_frames = deque(maxlen=fps*4)

condition1_met = condition2_met = condition3_met = False
saque_count = 0
post_condition3_frames = 0  # Novo contador para 2 segundos extras

current_frame = 0
# frame_skip = 2  # pular de 2 em 2 frames

while cap.isOpened():
    ret, frame = cap.read()
    # current_frame += 1
    # if not ret or current_frame % frame_skip != 0:
    #     continue
    
    scale_percent = 30  # percentual de escala
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    current_frame += 1
    completion_percentage = (current_frame / total_frames) * 100
    os.system("clear")
    print(f"Processamento: {completion_percentage:.2f}% concluído")

    # Processar a imagem e obter os resultados da pose
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    # for pose in results:
    #     print(pose)
    #     pose.landmark.pose += 1
    #     pose.landmark.pose_1 += 1
    #     if pose.landmark != pose_last:
    #         break


    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Desenhando landmarks no frame
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
            out = cv2.VideoWriter(f'{serve_dir}/{filename.split(".")[0]}_{saque_count}.mp4', fourcc, fps, (w, h))

        # Se a gravação começou, escreva os frames no arquivo
        if out is not None:            
            # Continue gravando por 2 segundos após a condição 3 ser verificada
            if post_condition3_frames >= fps * 2:
                with open(f'{land_dir}/landmarks_{filename.split(".")[0]}_{saque_count}.pickle', 'wb') as f:
                    pickle.dump(list(landmarks_frames), f)

                while not len(frames) == 0:
                    out.write(frames.popleft())

                out.release()
                out = None
                print(f"Saque {saque_count} salvo")
                # Resetando as condições e contadores para o próximo saque
                condition1_met = condition2_met = condition3_met = False
                post_condition3_frames = 0


    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
