from PIL import Image
import os

input_folder = ('/home/ali/Desktop/islenmis_cek_fotolari')

output_folder = '/home/ali/Desktop/New Folder 1'

new_size = (3000, 1200)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for idx, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        rgb_img = img.convert('RGB')

        # Görselleri yeniden boyutlandır
        resized_img = rgb_img.resize(new_size)

        # Yeni dosya adını oluştur
        new_filename = f'image_{idx + 1}.jpg'
        output_path = os.path.join(output_folder, new_filename)

        # Görselleri kaydet
        resized_img.save(output_path, 'JPEG')

print("Görseller başarıyla formatlandı, yeniden boyutlandırıldı ve yeniden adlandırıldı.")


