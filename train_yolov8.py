from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

# Modeli eğit
model.train(data='config.yaml', epochs=50, imgsz=640, batch=16)


####//** TEST İÇİN YAZILAN KODLAR **//####
# results = model.val()
#
# results = model.predict(source='cheques_dataset/images/val', save=True)
