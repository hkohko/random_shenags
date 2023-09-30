import easyocr
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import PurePath, Path
from constants import *
"""
credit: copacel on discord 
"""
# read image
image_path = 'C:/Users/Margelus/Downloads/Screenshot_18.png'

im = cv.imread(image_path)
imcopy = im.copy()
img_gray = cv.cvtColor(imcopy, cv.COLOR_BGR2GRAY)
topleft = (100, 15)
bottomright = (200, 40)
cv.rectangle(im, topleft, bottomright, (0,255,0),2)
plt.imshow(im)
plt.show()

roi = img_gray[topleft[1]: bottomright[1], topleft[0]:bottomright[0]]
plt.imshow(roi, "gray")
plt.show()

low = 170
max_num = 255
lower_thresh = tuple(low for _ in range(3))
max_thresh = tuple(max_num for _ in range(3))
_, thresh = cv.threshold(roi, low, max_num, cv.THRESH_BINARY_INV)
cv.imwrite(("img_dark_hopefully1.png"), thresh)
thresh_gray = cv.cvtColor(thresh, cv.COLOR_BGR2RGB)
plt.imshow(thresh_gray)
plt.show()


def zoom(img, zoom_factor):
    return cv.resize(img, None, fx=zoom_factor, fy=zoom_factor)

img_grey = cv.imread("img_dark_hopefully1.png")

zoomed_img_grey = zoom(img_grey,2)


# instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# detect text on image
text_ = reader.readtext(zoomed_img_grey)

threshold = 0.25
# draw bbox and text
for t_, t in enumerate(text_):
    print(t)

    bbox, text, score = t

    if score > threshold:
        #cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        #cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)
        ...

plt.imshow(cv.cvtColor(zoomed_img_grey, cv.COLOR_BGR2RGB))
plt.show()
