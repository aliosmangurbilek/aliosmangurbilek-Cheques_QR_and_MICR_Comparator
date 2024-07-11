from pylibdmtx.pylibdmtx import decode
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

def process_image(image_path):
    image = Image.open(image_path)


    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  #kontrast arrttır

    decoded_objects = decode(image)


    for obj in decoded_objects:
        print(f"DataMatrix kodu: {obj.data.decode('utf-8')}")

    if not decoded_objects:
        print("No DataMatrix code found.")

    plt.imshow(image)
    plt.title('Original Image')
    plt.show()
    plt.imshow(image)
    plt.title('Processed Image')
    plt.show()

if __name__ == "__main__":
    image_path = 'cheques/deniz_bank.jpg'
    process_image(image_path)

####okunanlar = garanti bankası, t-bank, vakıfbank, vakıfkatılım, ziraat, deniz, qnb1, yapıkredi
####preprocessing işlemi sonrasında emlak bank okundu öncesinde okunmamıştı

#---------------------------------------------------------#

# import cv2
# from pyzbar.pyzbar import decode
# import matplotlib.pyplot as plt

# def process_image(image_path):
#     # Görseli gri tonlamalı olarak yükleyin
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     # Görseli göster
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
#     plt.title('Original Image')
#     plt.show()
#
#     # Görselin kontrastını artırın
#     image = cv2.equalizeHist(image)
#
#     # İşlenmiş görseli göster
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
#     plt.title('Processed Image')
#     plt.show()
#
#     # DataMatrix kodlarını çözümleyin
#     decoded_objects = decode(image)
#
#     # DataMatrix kodlarını ve diğer barkodları ekrana yazdırın
#     for obj in decoded_objects:
#         print(f"Type: {obj.type}")
#         print(f"Data: {obj.data.decode('utf-8')}")
#         print(f"Position: {obj.rect}")
#
#     if not decoded_objects:
#         print("No DataMatrix code found.")
#
# if __name__ == "__main__":
#     image_path = 'cheques/garanti-0000.jpg'  # Görsel dosyanızın yolu
#     process_image(image_path)
###### hiç okunan yok

#-----------------------------------------------------------#

# import cv2
# import pyzxing
# import matplotlib.pyplot as plt
#
# def process_image(image_path):
#     # Görseli gri tonlamalı olarak yükleyin
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     if image is None:
#         print(f"Error: Unable to load image from {image_path}")
#         return
#
#
#
#     # Görselin kontrastını artırın
#     image = cv2.equalizeHist(image)
#
#
#     # İşlenmiş görseli geçici olarak kaydedin
#     processed_image_path = 'processed_image.png'
#     cv2.imwrite(processed_image_path, image)
#
#     # pyzxing kullanarak DataMatrix kodlarını çözümleyin
#     reader = pyzxing.BarCodeReader()
#     results = reader.decode(processed_image_path)
#
#     # pyzxing çıktısını kontrol edin
#     print("Decoding results:", results)
#
#     # Çözümlenen DataMatrix kodlarını ekrana yazdırın
#     if results:
#         for result in results:
#             if result.get('format') == 'DATA_MATRIX':
#                 print(f"DataMatrix kodu: {result.get('parsed')}")
#     else:
#         print("No DataMatrix code found.")
#
#     # Görseli göster
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
#     plt.title('Original Image')
#     plt.show()
#
#     # İşlenmiş görseli göster
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
#     plt.title('Processed Image')
#     plt.show()
#
# if __name__ == "__main__":
#     image_path = 'cheques/qnb_finansbank.jpg'  # Görsel dosyanızın yolu
#     process_image(image_path)
######## hiç okunan görsel yok

