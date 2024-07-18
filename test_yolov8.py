from ultralytics import YOLO
import os

model = YOLO('runs/detect/train3/weights/best.pt')  # En iyi model ağırlıklarını yükleyin

test_images_folder = 'cheques_dataset/images/val'

output_folder = 'cheques_dataset/results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

results = model.predict(source=test_images_folder, save=True, project=output_folder, name='predictions')

for result in results:
    print(f"Image: {result.path}")
    for box in result.boxes.xyxy:
        print(f"Box: {box[:4].tolist()}, Confidence: {box[4].item()}")
