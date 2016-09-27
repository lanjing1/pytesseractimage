#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

# Accessing and Modifying pixel values
img = cv2.imread("src/image/drivinglicense1.png")
# accessing only blue pixel

blue = img[100, 100, 0]
# accessing Red value
img.item(10,10,2)
# modifying Red value
img.itemset((10,10,2),100)
# It returns a tuple of number of rows,columns and channels(if image is color)
# print img.shape
# total number of pixels is accessed by img.size
# print img.size
# print img.dtype
ball = img[100:120, 160:180]
img[50:70,80:100] = ball
# cv2.imshow("rect",img)
# cv2.waitKey(0)
# the B,G,R channels of an image can be split into their individual planes when needed
b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))

BLUE = [255,0,0]
img1 = cv2.imread("OpenCVlogo.jpg")
replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
#plt.show()

# Image addition
x = np.uint8([250])
y = np.uint8([10])
# print cv2.add(x,y) # 250+10 = 260 => 255
# print x+y          # 250+10 = 260 % 256 = 4


# Image Blending
# img = np.zeros((220, 237, 3), np.uint8)
# print img.shape
# print img1.shape
# cv2.imshow("image",img)
# cv2.waitKey(0)
# dst = cv2.addWeighted(img,0.7,img1,0.3,0)


# Bitwise Operations
img1 = cv2.imread("test.jpg")
img2 = cv2.imread("OpenCVlogo.jpg")
# put logo on top-left corner,So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows,0:cols]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
cv2.imshow("img2gray",img2gray)
cv2.waitKey(0)
# 将灰度图二值化以便更清楚的观察结果
ret, mask = cv2.threshold(img2gray,150,255,cv2.THRESH_BINARY)
cv2.imshow("mask",mask)
cv2.waitKey(0)
# 反色，对二值图每个像素取反
mask_inv = cv2.bitwise_not(mask)
cv2.imshow("mask_inv",mask_inv)
cv2.waitKey(0)
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
cv2.imshow("img1_bg",img1_bg)
cv2.waitKey(0)
# Take only region of logo from logo image

img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
cv2.imshow("img2_bg",img1_bg)
cv2.waitKey(0)
# put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
cv2.imshow("dst",dst)
cv2.waitKey(0)
img1[0:rows, 0:cols] = dst
cv2.imshow("img2_fg",img2_fg)
cv2.waitKey(0)
cv2.imshow("res",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.getTickCount() # returns the number of clock-cycles after a reference event
cv2.getTickFrequency() # returns the frequency of clock-cycles,or the number of clock-cycles per second
cv2.useOptimized() # check if optimization is enabled