import torch
import cv2
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# URL do stream RTSP da câmera IP
rtsp_url = "rtsp://admin:*HXxiwj8@10.0.0.208:554/0.sdp?real_stream"

# Capturar a imagem da câmera IP
cap = cv2.VideoCapture(rtsp_url)
ret, frame = cap.read()
cap.release()

# Verificar se a captura foi bem-sucedida
if not ret:
    print("Falha ao capturar a imagem da câmera IP")
    exit()

# Converter a imagem de BGR (OpenCV) para RGB (PIL)
image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

device = "cuda"
model_repo = "MarianaMCruz/detr-finetuned-ppe"

image_processor = AutoImageProcessor.from_pretrained(model_repo)
model = AutoModelForObjectDetection.from_pretrained(model_repo)
model = model.to(device)

with torch.no_grad():
    inputs = image_processor(images=[image], return_tensors="pt")
    outputs = model(**inputs.to(device))
    target_sizes = torch.tensor([[image.size[1], image.size[0]]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.55, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
    
draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x, y, x2, y2 = tuple(box)
    draw.rectangle((x, y, x2, y2), outline="red", width=1)
    draw.text((x, y), model.config.id2label[label.item()], fill="white")

image.show()
