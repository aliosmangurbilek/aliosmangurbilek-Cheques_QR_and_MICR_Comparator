# import easyocr
#
# # OCR okuyucusunu oluşturun
# reader = easyocr.Reader(['en'])  # 'en' yerine MICR kodlar için 'latin' dilini kullanabilirsiniz
#
# # Çek görüntüsünü yükleyin
# image_path = 'vakıf_katılım_cek.jpg'
#
# # Metni çıkarın
# result = reader.readtext(image_path)
#
# # Sonuçları yazdırın
# for (bbox, text, prob) in result:
#     print(f"Bulunan metin: {text}, Doğruluk: {prob}")

#--------------------------------------------------------------------#

# from google.cloud import vision
# import io
#
# # Google Cloud Vision API istemcisini oluşturun
# client = vision.ImageAnnotatorClient()
#
# # Çek görüntüsünü yükleyin
# image_path = 'vakıfbank_cek.jpg'
# with io.open(image_path, 'rb') as image_file:
#     content = image_file.read()
#
# image = vision.Image(content=content)
#
# # Metni tespit edin
# response = client.text_detection(image=image)
# texts = response.text_annotations
#
# # Sonuçları yazdırın
# for text in texts:
#     print(f"Bulunan metin: {text.description}")

#----------------------------------------------------#

# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
#
# # Model ve işlemciyi yükleyin
# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
#
# # Çek görüntüsünü yükleyin
# image_path = 'cheques/garanti-0000.jpg'
# image = Image.open(image_path)
#
# # Metni çıkarın
# pixel_values = processor(image, return_tensors="pt").pixel_values
# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
#
# print(f"Bulunan metin: {generated_text}")

import cv2
import pytesseract

# Resmi yükle
image = cv2.imread('cheques/vakıfbank_cek.jpg')

# Gri tonlamaya çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gürültüyü azaltmak ve algılamayı iyileştirmek için GaussianBlur uygula
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding uygula
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Konturları bul
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Orijinal resim üzerinde konturları çiz
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Tesseract'ı E13B diliyle kullanarak OCR işlemi yap
custom_config = r'--oem 3 --psm 6'
micr_text = pytesseract.image_to_string(thresh, config=custom_config, lang='e13b')

print("Tespit Edilen MICR metni:", micr_text)

# Konturlarla resmi göster
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()




