import json
import os

json_folder = 'cheques_json_labeled'
image_folder = 'cheques_dataset'
output_folder = 'cheques_txt'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for json_file in os.listdir(json_folder):
    if json_file.endswith('.json'):
        with open(os.path.join(json_folder, json_file)) as f:
            data = json.load(f)

        image_name = data['imagePath'].split('/')[-1]

        boxes = data['shapes']

        image_width = 2481
        image_height = 1181

        yolo_labels = []
        for box in boxes:
            if box['label'] == 'micr_code':
                class_id = 0  # MICR kodu için sınıf etiketi
                x_min, y_min = box['points'][0]
                x_max, y_max = box['points'][1]
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height
                yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

        # Etiket dosyasını kaydet
        with open(os.path.join(output_folder, os.path.splitext(image_name)[0] + '.txt'), 'w') as f:
            f.write('\n'.join(yolo_labels))
