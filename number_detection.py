import pytesseract
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from imutils import contours
import imutils
import os
import re
import pandas as pd
from csv import DictWriter

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):
    charIter = charCnts.__iter__()
    rois = []
    locs = []

    while True:
        try:
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None

            if cW >= minW and cH >= minH:
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))
            else:
                parts = [c, next(charIter), next(charIter)]
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf, -np.inf)

                for p in parts:
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)

                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))
        except StopIteration:
            break

    return (rois, locs)


charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
             "T", "U", "A", "D"]

ref_image_path = 'cheques/micr_e13b_reference.png'  # MICR kodlarının referans olduğu görsel

if not os.path.exists(ref_image_path):
    print(f"Error: The file {ref_image_path} does not exist.")
    exit()

ref = cv2.imread(ref_image_path)
if ref is None:
    print(f"Error: Unable to read the image at {ref_image_path}.")
    exit()

plt.imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
plt.title('Reference Image')
# plt.show()

ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
plt.imshow(ref, cmap='gray')
plt.title('Grayscale Reference Image')
# plt.show()

ref = imutils.resize(ref, width=400)
plt.imshow(ref, cmap='gray')
plt.title('Resized Reference Image')
# plt.show()

ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
plt.imshow(ref, cmap='gray')
plt.title('Thresholded Reference Image')
# plt.show()

refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]

if len(refCnts) == 0:
    print("Error: No contours found in the reference image.")
    exit()

# Kontur kontrolü
for i, c in enumerate(refCnts):
    if not isinstance(c, np.ndarray):
        raise TypeError(f"Kontur {i} numpy array değil, tipi: {type(c)}")
    elif len(c) == 0:
        raise ValueError(f"Kontur {i} boş")

refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
(refROIs, refLocs) = extract_digits_and_symbols(ref, refCnts, minW=10, minH=20)
chars = {}

for (name, roi, loc) in zip(charNames, refROIs, refLocs):
    roi = cv2.resize(roi, (36, 36))
    chars[name] = roi

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
output = []

fileNames = ['cheques/vakıfbank_cek.jpg']  # Çekin fotoğrafının yolu

inputFile = fileNames[0]
if not os.path.exists(inputFile):
    print(f"Error: The file {inputFile} does not exist.")
    exit()

image = cv2.imread(inputFile)
if image is None:
    print(f"Error: Unable to read the image at {inputFile}.")
    exit()

(h, w,) = image.shape[:2]
delta = int(h - (h * 0.2))
bottom = image[delta:h, 0:w]

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

plt.imshow(cv2.cvtColor(bottom, cv2.COLOR_BGR2RGB))
plt.title('Bottom Part of Image')
plt.show()

gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

plt.imshow(blackhat, cmap='gray')
plt.title('Blackhat')
plt.show()

gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=7)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

plt.imshow(gradX, cmap='gray')
plt.title('Gradient')
plt.show()

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

plt.imshow(thresh, cmap='gray')
plt.title('Threshold')
plt.show()

thresh = clear_border(thresh)

plt.imshow(thresh, cmap='gray')
plt.title('Threshold without Border')
plt.show()

groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
groupCnts = imutils.grab_contours(groupCnts)
groupLocs = []
for (i, c) in enumerate(groupCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    if w > 5 and h > 5:
        groupLocs.append((x, y, w, h))

groupLocs = sorted(groupLocs, key=lambda x: x[0])

for (gX, gY, gW, gH) in groupLocs:
    groupOutput = []
    group = gray[gY - 15:gY + gH + 15, gX - 15:gX + gW + 15]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    charCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    charCnts = imutils.grab_contours(charCnts)

    # Kontur kontrolü
    for i, c in enumerate(charCnts):
        if not isinstance(c, np.ndarray):
            raise TypeError(f"Kontur {i} numpy array değil, tipi: {type(c)}")
        elif len(c) == 0:
            raise ValueError(f"Kontur {i} boş")

    charCnts = contours.sort_contours(charCnts, method="left-to-right")[0]
    (rois, locs) = extract_digits_and_symbols(group, charCnts)

    for roi in rois:
        scores = []
        roi = cv2.resize(roi, (36, 36))
        for charName in charNames:
            result = cv2.matchTemplate(roi, chars[charName], cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        groupOutput.append(charNames[np.argmax(scores)])

    cv2.rectangle(image, (gX - 10, gY + delta - 10), (gX + gW + 10, gY + gY + delta), (0, 0, 255), 2)
    cv2.putText(image, "".join(groupOutput), (gX - 10, gY + delta - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
    output.append("".join(groupOutput))

print("Check OCR: {}".format(" ".join(output)))

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Output Image')
plt.show()

output = ''.join(output)
chequeNum = ''
idx = output[1:].find(output[0])

chequeNum = output[:8]
micrCode = output[8:]
print(chequeNum)
print(micrCode)

bgSettings = "blur"  # "thresh" or "blur"
if bgSettings == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif bgSettings == "blur":
    gray = cv2.medianBlur(gray, 3)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename), lang="eng")
os.remove(filename)
print(text)

lines = text.split('\n')

accountNum = ''
ifscCode = ''
regexp = re.compile('IFS')
for line in lines[:-4]:
    num = re.findall(r'\d+', line)
    num = ''.join(num)
    if len(num) > 12:
        accountNum = num
        break

for line in lines:
    if regexp.search(line):
        segments = line.split(' ')
        code = re.findall(r'\d+', line)
        regexp1 = re.compile(code[0])
        for seg in segments:
            if regexp1.search(seg):
                ifscCode = seg
                break
        break

print(ifscCode)
print(accountNum)

headersCSV = ['CHEQUE-NUMBER', 'MICR-CODE', 'IFSC-CODE', 'ACCOUNT-NUMBER']
dict = {'CHEQUE-NUMBER': chequeNum, 'MICR-CODE': micrCode, 'IFSC-CODE': ifscCode, 'ACCOUNT-NUMBER': accountNum}

csv_file = 'chequeData.csv'
if not os.path.isfile(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = DictWriter(f, fieldnames=headersCSV)
        writer.writeheader()

with open(csv_file, 'a', newline='') as f:
    dictwriter_object = DictWriter(f, fieldnames=headersCSV)
    dictwriter_object.writerow(dict)

data = pd.read_csv(csv_file)
print(data)
