from ultralytics import YOLO

model = YOLO('yolov8l.yaml')

hyperparameters = {
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 0.05,
    'cls': 0.5,
    'dfl': 1.5,
    'pose': 12.0,
    'label_smoothing': 0.0,
    'nbs': 64,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0
}

model.train(
    data='/home/ali/PycharmProjects/aliosmangurbilek-Cheques_QR_and_MICR_Comparator/cheques_dataset/config.yaml',
    epochs=150,
    imgsz=2481,
    batch=2,
    lr0=0.0005,
    augment=True,
    **hyperparameters
)



