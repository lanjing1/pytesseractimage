#!/usr/bin/python3
# -*- coding: utf8 -*-

# learn how to convert images from one color-space to another,like BGR<->Gray,BGR<->HSV etc

import cv2
import numpy as np
from matplotlib import pyplot as plt

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print flags

green = np.uint8([[[0,255,0]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print hsv_green

# Simple Thresholding
def sim_threshold():
    img = cv2.imread('src/image/61.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img_gray",img_gray)
    cv2.waitKey(0)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)


    plt.imshow(img_gray, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
    plt.show()
    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

    for i in xrange(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# Adaptive Thresholding
  # Adaptive Method-It decides how thresholding value is calculated
    # cv2.ADAPTIVE_THRESH_MEAN_C: threshold value is the mean of neighbourhood area
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C: threshold value is the weighted sum of
    # neighbourhood values where weights are a gaussian window
  # Block Size -It decides the size of neighbourhood area
  # C-It is just a constant which is subtracted from the mean or weighted mean calculated
def adap_threshold():
    img = cv2.imread('src/image/61.jpg',0)
    img = cv2.medianBlur(img,5)

    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in xrange(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    plt.show()

# Otsu's Binarization
 # bimodal image is an image whose histogram has two peaks
 # Approximately take a value in the middle of those peaks as threshold value

def ostu_binar():
    img = cv2.imread("src/image/61.jpg",0)
    # global thresholding
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
    for i in xrange(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()

# Geometric Transformations of Images
def geo_trans():
    # Scalling
    img = cv2.imread("src/image/61.jpg")
    img_resize = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    height,width = img.shape[:2]
    res = cv2.resize(img, (220, 330), interpolation=cv2.INTER_CUBIC)

    # Translation is the shifting of object's location
    rows,cols,pix = img.shape
    M = np.float32([[1,0,10],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))

    # Rotation of an image for an angle  按等比例以矩心为中心旋转90度
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dst = cv2.warpAffine(img,M,(cols,rows))

    # Affine Transformation-仿射转换
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))

    # Perspective Transformation-透视变换
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.imshow("img", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Smoothing Images
#  Blur imagess with various low pass filters Apply custom-made filters to images (2D convolution)

def smooth_image():
    # 2D Convolution(Image Filtering)--二维卷积
    # As for one-dimensional signals, images also can be filtered with various low-pass filters (LPF),
    # high-pass filters (HPF),etc. A LPF helps in removing noise, or blurring the image.
    #  A HPF filters helps in finding edges in an image.
    img = cv2.imread("OpenCVlogo.jpg")
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    # adap_threshold()
    # ostu_binar()
    # geo_trans()
    smooth_image()