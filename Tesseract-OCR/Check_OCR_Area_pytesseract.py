import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt


# 讀取圖像
image = cv2.imread('./123456.jpeg')

# 轉換灰階圖像 
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Tesseract進行辨識 
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\10410056\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# -c tessedit_char_blacklist 把指定字元Ban掉，不辨識
config = r'-c tessedit_char_blacklist=OTlIi --oem 3 --psm 6'
boxes_data  = pytesseract.image_to_boxes(gray_image, lang = 'eng',output_type=pytesseract.Output.STRING, config=config) #取得boxes大小 

threshold_value = 150
def getPuTextSize(img):
        sz = min(img.shape[:2]) / 500  # 影像短邊 / 500
        lw = max(round(sz), 1)
        font_sz = max(sz / 3, 1)
        return lw, font_sz

def OCR_Area_Fuction(image:np.ndarray,gray_image:np.ndarray,boxes_data:str) -> str :
    height, width, _ = image.shape
    ocr_str = ''
    OCR_Area_list =[] #建立OCR_Area_list 等用來儲存OCR Area用
    for box in boxes_data.splitlines():
        b = box.split()
        #取出字元，位置x,y，長高
        char, x, y, w, h = b[0], int(b[1]), int(b[2]), int(b[3]), int(b[4])
        roi = gray_image[height - h:height - y, x:w]
        # 進行二值化
        _, binary_roi = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY)
        # 統計黑色像素
        black_pixel_count = roi.size - cv2.countNonZero(binary_roi)
        OCR_Area_dit = { char : black_pixel_count }
        lw , font = getPuTextSize(roi)
        # 繪製矩形框
        cv2.rectangle(image, (x, height - y), (w, height - h), (0, 255, 0), 2)
        cv2.putText(image, char, (x, height- int(y)), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 0, 0), lw, cv2.LINE_AA)
        ocr_str += char
        OCR_Area_list.append(OCR_Area_dit)
    return ocr_str, OCR_Area_list
ocr_str, OCR_Area_list = OCR_Area_Fuction(image,gray_image,boxes_data)
print(f'OCR_char : {ocr_str}')
print(f'OCR_Area_list : {OCR_Area_list}')
cv2.imshow('OCR_Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
