import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import time
import threading
import queue
import sys
import json

with open('config/camera_settings.json') as config_file:
    config = json.load(config_file)

# URL do stream RTSP da câmera IP
rtsp_url = config['CAMERA1_KEY']

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(rtsp_url)

# Verificar se a captura foi bem-sucedida
if not cap.isOpened():
    print("Falha ao conectar à câmera IP")
    exit()

device = "cuda"
model_repo = "MarianaMCruz/detr-finetuned-ppe"

image_processor = AutoImageProcessor.from_pretrained(model_repo)
model = AutoModelForObjectDetection.from_pretrained(model_repo)
model = model.to(device)

# Definir a proporção desejada (por exemplo, 50% do tamanho original)
scale_percent = 50

# Fila para armazenar o frame mais recente
frame_queue = queue.Queue(maxsize=1)

# Evento para sinalizar o encerramento das threads
stop_event = threading.Event()

# Inicializar o VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 5.0, (int(cap.get(3) * scale_percent / 100), int(cap.get(4) * scale_percent / 100)))

def capture_frames():
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame")
            break
        if not frame_queue.empty():
            frame_queue.get()  # Descarta o frame antigo
        frame_queue.put(frame)

def process_frames():
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            start_time = time.time()

            # Redimensionar o frame mantendo a proporção
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            # Converter a imagem de BGR (OpenCV) para RGB (PIL)
            image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

            with torch.no_grad():
                inputs = image_processor(images=[image], return_tensors="pt")
                outputs = model(**inputs.to(device))
                target_sizes = torch.tensor([[image.size[1], image.size[0]]])
                results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                x, y, x2, y2 = tuple(box)
                cv2.rectangle(resized_frame, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                # Formatar a label com o score
                label_with_score = f"{model.config.id2label[label.item()]}: {score:.2f}"
                cv2.putText(resized_frame, label_with_score, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
            out.write(resized_frame)

            # Exibir o frame redimensionado com as bounding boxes
            cv2.imshow('Object Detection in Real Time', resized_frame)

            # Pressione 'q' para sair do loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            # Calcular o tempo de processamento e ajustar a captura
            processing_time = time.time() - start_time
            time.sleep(max(0, 1.0 - processing_time))

# Criar e iniciar as threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()

# Esperar até que a janela seja fechada
window_opened = False
while not stop_event.is_set():
    if cv2.getWindowProperty('Object Detection in Real Time', cv2.WND_PROP_VISIBLE) >= 1:
        window_opened = True
    if window_opened and cv2.getWindowProperty('Object Detection in Real Time', cv2.WND_PROP_VISIBLE) < 1:
        stop_event.set()

capture_thread.join()
process_thread.join()

# Liberar a captura e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()
sys.exit()