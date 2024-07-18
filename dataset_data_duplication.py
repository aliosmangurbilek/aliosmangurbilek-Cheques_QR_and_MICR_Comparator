import os
import shutil
import random
from PIL import Image, ImageEnhance
from torchvision import transforms

# Veri artırma dönüşümleri
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(640, scale=(0.8, 1.0)),
])


def augment_image(image):
    # Veri artırma dönüşümlerini uygula
    image = augmentation_transforms(image)
    return image


def duplicate_and_augment_images(source_dir, target_dir, num_copies):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if os.path.isfile(file_path):
            image = Image.open(file_path)
            for i in range(num_copies):
                augmented_image = augment_image(image)
                new_filename = f"{os.path.splitext(filename)[0]}_copy{i}{os.path.splitext(filename)[1]}"
                new_file_path = os.path.join(target_dir, new_filename)
                augmented_image.save(new_file_path)


# Örnek kullanım (her görüntü için 3 kopya oluşturma)
duplicate_and_augment_images('cheques_dataset/images/train', 'cheques_dataset/images/train', 3)
