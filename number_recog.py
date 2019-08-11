import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import tesserocr
# import PIL.Image
import tesseract
import os
TESSERACT_DIR = 'D:\Program Files\Tesseract-OCR'
temp = os.environ['Path']
os.environ['Path'] = f'{TESSERACT_DIR};' + os.environ['Path']
tess = tesseract.Tesseract(language=b'eng', datapath=bytes(f'{TESSERACT_DIR}\\tessdata', 'utf-8'), lib_path=f'{TESSERACT_DIR}\libtesseract-5.dll')
os.environ['Path'] = temp


def number_recog(roi):
    """
    Image size of 720p is required.
    """
    number_area = roi[85:115, 45:115, :]
    number_area_gray = cv2.cvtColor(number_area, cv2.COLOR_BGR2GRAY)
    threshed = cv2.adaptiveThreshold(number_area_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, -45)
    contours, _ = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_box = []
    for ct in contours:
        if cv2.contourArea(ct) > 10:
            x, y, w, h = cv2.boundingRect(ct)
            if 10 / 45 < w / h and w / h < 35 / 45 and 12 < h and h < 18 and abs(y + h // 2 - 15) < 5:
                valid_box.append((x, y, w, h))
    valid_box = sorted(valid_box, key=lambda x: x[0])
    valid_box_2 = []
    for box in valid_box:
        x, y, w, h = box
        if len(valid_box_2) == 0 or (abs(x - valid_box_2[-1][0]) > 7 and abs(x - valid_box_2[-1][0]) < 15):
            valid_box_2.append(box)
    number = 0
    for box in valid_box_2:
        x, y, w, h = box
        digit_image = threshed[y:y + h, x:x + w]
        digit_image = np.pad(digit_image, ((3, 3), (3, 3)), mode='constant')
        digit = tesseract.tesseract_process_image2(tess, tesseract.FramePiece(digit_image, b'0123456789', 10, 0))
        # with tesserocr.PyTessBaseAPI(path='D:/Program Files/Tesseract-OCR/tessdata/', lang='eng', psm=tesserocr.PSM.SINGLE_CHAR, oem=tesserocr.OEM.TESSERACT_ONLY) as api:
        #     api.SetVariable('tessedit_char_whitelist', '0123456789')
        #     api.SetImage(PIL.Image.fromarray(digit_image))
        #     text = api.GetUTF8Text().strip()
        #     digit = int(text)
        number = number * 10 + int(digit) if digit != b'' else 0
    return number
