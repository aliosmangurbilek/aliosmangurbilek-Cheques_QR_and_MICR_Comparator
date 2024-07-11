import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV dosyasını yükleme
file_path = 'E13B.csv'  # Dosyanızın yolunu buraya girin
data = pd.read_csv(file_path)

# Mevcut etiketleri ve dağılımlarını kontrol etme
label_counts = data['m_label'].value_counts()
print("Label distribution in the dataset:")
print(label_counts)

# Görselleri hazırlama
image_height, image_width = 20, 20  # Görüntü boyutları
label_to_visualize = 5  # Görselleştirmek istediğiniz etiket
num_images_to_show = 5  # Görselleştirilecek örnek sayısı

# Belirli bir etikete sahip görselleri filtreleme
specific_label_data = data[data['m_label'] == label_to_visualize].iloc[:num_images_to_show]

if specific_label_data.empty:
    print(f"No data found for label {label_to_visualize}")
else:
    # Görselleri hazırlama
    images = specific_label_data.iloc[:, 8:].values
    images = np.array([np.pad(x, (0, image_height*image_width - len(x)), 'constant') for x in images])
    images = images.reshape(-1, image_height, image_width)

    # Görselleri çizdirme
    fig, axes = plt.subplots(1, num_images_to_show, figsize=(15, 5))
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f"Label: {label_to_visualize}")
        else:
            ax.axis('off')
            ax.set_title("No more images")

    plt.show()
