import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input


# CSV dosyasını yükleme
file_path = 'E13B.csv'
data = pd.read_csv(file_path)

# Veri setindeki her bir satırın piksel sayısını kontrol etme
num_pixels = data.iloc[:, 8:].shape[1]
print(f"Each image has {num_pixels} pixels.")

# Piksel verilerini ve etiketleri ayırma
X = data.iloc[:, 8:].values  # Piksel verileri sütunları
y = data['m_label'].values  # Etiketler

# Piksel verilerini yeniden şekillendirme ve normalleştirme
image_height = 21
image_width = 21
# Negative padding değerlerinden kaçınmak için padding işlemini dikkatli yapma
if image_height * image_width >= num_pixels:
    X = np.array([np.pad(x, (0, image_height*image_width - len(x)), 'constant') for x in X])  # Piksel verilerini padding ile tam kare yapma
else:
    raise ValueError("Görüntü boyutları belirlenen image_height ve image_width ile uyumlu değil.")
X = X.reshape(-1, image_height, image_width, 1)  # (örnek sayısı, yükseklik, genişlik, kanal sayısı)
X = X / 255.0  # Normalleştirme

# Etiketleri one-hot encoding ile dönüştürme
y = to_categorical(y)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma
model = Sequential([
    Input(shape=(image_height, image_width, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

# Modeli derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Modeli değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
