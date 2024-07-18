import os
import shutil
import random

image_folder = 'cheques_dataset'
label_folder = 'cheques_txt'

train_image_folder = 'cheques_dataset/images/train'
val_image_folder = 'cheques_dataset/images/val'
train_label_folder = 'cheques_dataset/labels/train'
val_label_folder = 'cheques_dataset/labels/val'

os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(val_image_folder, exist_ok=True)
os.makedirs(train_label_folder, exist_ok=True)
os.makedirs(val_label_folder, exist_ok=True)

images = [f for f in os.listdir(image_folder) if f.endswith(('.jpeg', '.jpg', '.png'))]
labels = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

data = list(zip(images, labels))
random.shuffle(data)

train_ratio = 0.8
train_size = int(len(data) * train_ratio)

train_data = data[:train_size]
val_data = data[train_size:]

for image, label in train_data:
    shutil.copy(os.path.join(image_folder, image), os.path.join(train_image_folder, image))
    shutil.copy(os.path.join(label_folder, label), os.path.join(train_label_folder, label))

for image, label in val_data:
    shutil.copy(os.path.join(image_folder, image), os.path.join(val_image_folder, image))
    shutil.copy(os.path.join(label_folder, label), os.path.join(val_label_folder, label))

print("Veri seti başarıyla yapılandırıldı.")
