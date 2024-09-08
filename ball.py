import cv2
import numpy as np
from collections import deque

# Abrir o vídeo
diff = deque([2])
cap = cv2.VideoCapture('kyrgios.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para espaço de cores HSV
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir intervalo da cor amarela
    amarelo_baixo = np.array([20, 50, 166])  # ajuste estes valores
    amarelo_alto = np.array([40, 100, 255])


    # Máscara para capturar áreas de cor amarela
    mascara = cv2.inRange(hsv, amarelo_baixo, amarelo_alto)
    diff.appendleft(mascara)
    # Encontrar contornos
    if len(diff) > 1:
        mask2 = diff[1] - diff[0]
        contornos, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            # Aproximar o contorno a um círculo e desenhá-lo
            (x, y), raio = cv2.minEnclosingCircle(contorno)
            centro = (int(x), int(y))
            raio = int(raio)
            if raio > 15 and raio < 25: # Ajuste este valor conforme necessário
                cv2.circle(frame, centro, raio, (0, 255, 0), 2)

        # Exibir o quadro
        cv2.imshow('Detecção da Bolinha de Tênis', frame)
        # cv2.imshow('Detecção da Bolinha de Tênis', mascara)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()