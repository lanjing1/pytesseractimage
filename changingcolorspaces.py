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
    img = cv2.imread("noiseLogo.png")
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)
    # plt.subplot(121), plt.imshow(img), plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    # Image Blurring(Image Smoothing)
    # 1.Averaging
    blur = cv2.blur(img,(5,5))

    # 2.Gaussian Filtering is highly effective in removing Gaussian noise from the image
    gaussianBlur = cv2.GaussianBlur(img,(5,5),0)

    # ???3.Median Filtering is highly effective in removing salt-and-pepper noise
    median = cv2.medianBlur(img,5)

    # 4.Biateral Filtering 双边过滤器 is highly effective at noise removal while preserving edges.
    # But the operation is slower compared to other filters.
    bilateralblur = cv2.bilateralFilter(img,9,75,75)
    cv2.imshow("blur",blur)
    cv2.waitKey(0)
    cv2.imshow("gaussianBlur", gaussianBlur)
    cv2.waitKey(0)
    cv2.imshow("median", median)
    cv2.waitKey(0)
    cv2.imshow("bilateralblur", bilateralblur)
    cv2.waitKey(0)

    # Morphological operations 形态学操作
    # like Erosion,Dilation,Opening,Closing etc
def morphological_tran():
    # erosion腐蚀
    img = cv2.imread("j.png",0)
    kernel = np.ones((8,8),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    #cv2.imshow("erosion", erosion)
    #cv2.waitKey(0)

    # Dilation 膨胀
    dilation = cv2.dilate(img,kernel,iterations=1)
    #cv2.imshow("dilation", dilation)
    #cv2.waitKey(0)

    # Opening is just another name of erosion followed by dilation.It is useful in removing noise
    img = cv2.imread("jnoise.png", 0)
    opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

    # Closing is reverse of Opening,Dilation followed by erosion.It is useful in closing small holes inside the foreground object,or small black points on the object
    # ???? The result will look like the outline of the object
    img = cv2.imread("jinnerNoise.png",0)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("closing", closing)
    #cv2.waitKey(0)

    # Morphological Gradient It is the difference between dilation and erosion of an image
    gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)

    # ???? Top Hat It is the difference between input image and Opening of the image
    tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)

    # ???? Black Hat It is the difference between the closing of the input image and input image
    blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.imshow("tophat", tophat)
    cv2.waitKey(0)

    # Structuring Element just pass the shape and size of the kernel,you get the desired kernel
    # Rectangular Kernel
    cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # Elliptical Kernel
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # Cross-shaped Kernel
    cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

    # Image Gradients

    # Find Image gradients,edges etc
def image_gradients():
    # Sobel and Scharr Derivatives

    # Laplacian Derivatives

    # ??? all operator in a single diagram
    img = cv2.imread('gezi.png', 0)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    # plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # Important matter 结果棱角模糊
    img = cv2.imread('box.png', 0)
    # Output dtype = cv2.CV_8U
    sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)
    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(sobelx8u, cmap='gray')
    plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(sobel_8u, cmap='gray')
    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Concept of Canny edge detection
def canny_edge():
    img = cv2.imread('gezi.png', 0)
    edges = cv2.Canny(img, 50, 100)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # ??? 报错Image Pyramids


if __name__ == '__main__':
    def img_pyramids():
        img = cv2.imread('gezi.png')
        G = img.copy()
        #res = cv2.resize(img,(300,300))
        lower_reso1 = cv2.pyrDown(G)
        higher_reso2 = cv2.pyrUp(lower_reso1)
        # cv2.imshow("lower_reso",lower_reso1)
        # cv2.waitKey(0)

        lower_reso2 = cv2.pyrDown(lower_reso1)
        # cv2.imshow("lower_reso2", lower_reso2)
        # cv2.waitKey(0)

        lower_reso3 = cv2.pyrDown(lower_reso2)
        # cv2.imshow("lower_reso3", lower_reso3)
        # cv2.waitKey(0)

        lower_reso4 = cv2.pyrDown(lower_reso3)
        # cv2.imshow("lower_reso4", lower_reso4)
        # cv2.waitKey(0)

        # Image Blending using Pyramids
        A = cv2.imread('apple.png')
        B = cv2.imread('orange.png')
        # generate Gaussian pyramid for A
        G = A.copy()
        gpA = [G]
        for i in xrange(6):
            G = cv2.pyrDown(G)
            gpA.append(G)

        # generate Gaussian pyramid for B
        G = B.copy()
        gpB = [G]

        for i in xrange(6):
            G = cv2.pyrDown(G)
            gpB.append(G)
        # generate Laplacian Pyramid for A
        lpA = [gpA[5]]
        for i in xrange(5, 0, -1):
            GE = cv2.pyrUp(gpA[i])
            L = cv2.subtract(gpA[i-1], GE)
            lpA.append(L)
        # generate Laplacian Pyramid for B
        lpB = [gpB[5]]
        for i in xrange(5, 0, -1):
            GE = cv2.pyrUp(gpB[i])
            L = cv2.subtract(gpB[i - 1], GE)
            lpB.append(L)
        # Now add left and right halves of images in each level
        LS = []
        for la, lb in zip(lpA, lpB):
            rows, cols, dpt = la.shape
            ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
            LS.append(ls)
        # now reconstruct
        ls_ = LS[0]
        for i in xrange(1, 6):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[i])
        # image with direct connecting each half
        real = np.hstack((A[:, :cols / 2], B[:, cols / 2:]))
        cv2.imwrite('Pyramid_blending2.jpg', ls_)
        cv2.imwrite('Direct_blending.jpg', real)

        # Contours in OpenCV
        # The contours are a useful tool for shape analysis and object detection and recognition.
        # 1 For better accuracy,use binary images.So before finding contours,apply threshold or canny edge detection
        # 2 findContours function modifies the source image
        # 3 In OpenCV,finding contours is like finding white object from black ground.


def contours():
    img = cv2.imread("j.png")
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 300, 600, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ???只保留线的end points cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    # 1.Moments help to calculate some features like center of mass of the object,area of the object etc
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    print M

    # 2.Contour Area
    area = cv2.contourArea(cnt)

    # 3. Contour Perimeter
    perimeter = cv2.arcLength(cnt, True)

    # 4.Contour Approximation
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cont_img = cv2.drawContours(img, approx, -1, (0, 255, 0), 3)

    # 5.Convex Hull
    hull = cv2.convexHull(cnt)
    cont_img = cv2.drawContours(img, hull, -1, (0, 255, 0), 3)

    # 6.Checking Convexity
    k = cv2.isContourConvex(cnt)

    # 7.Bounding Rectangle
    # 7.a. Straight Bounding Rectangle
    x, y, w, h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 7.b. Rotated Rectangle
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # 8. Minimum Enclosing Circle
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    img = cv2.circle(img, center, radius, (0, 255, 0), 2)

    # 9. Fitting an Ellipse
    ellipse = cv2.fitEllipse(cnt)
    img = cv2.ellipse(img, ellipse, (0, 255, 0), 2)

    # 10. Fitting a Line
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    img = cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
    #............


    # 直方图均值化Histograms Equalization in OpenCV.Even if the image was a darker image,
    # 重点after equalization we will get almost the same image as we got.
def histograms():
    img = cv2.imread('card.jpg', 0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))  # stacking images side-by-side
    cv2.imwrite('rescard.png', res)
    cv2.imwrite('equ.png', equ)
    img = cv2.imread('zqqtes.png', 0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))  # stacking images side-by-side
    cv2.imwrite('zqqtesaa.png', equ)
    cv2.imwrite('zqqtesaad.png', res)

    # 1.4.12 Template Matching
    # to finds objects in an image using Template Matching
    # 重点
def temp_match():
    img = cv2.imread("src/image/receipt1.jpg",0)
    img2 = img.copy();
    template = cv2.imread('template.png',0)
    w,h = template.shape[::-1]
    # all the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # apply template matching
        res = cv2.matchTemplate(img,template,method)
        min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

        # if the method is TM_SQDIFF or TM_SQDIFF_NORMED,take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left,bottom_right,(255,0,0),2)
        # 购买方名称及身份证号码
        cv2.rectangle(img,(top_left[0]+163,top_left[1]+176),(top_left[0]+536,top_left[1]+245),(255,0,0),2)
        # 厂牌型号
        cv2.rectangle(img,(top_left[0]+496,top_left[1]+247),(top_left[0]+858,top_left[1]+294),(255,0,0),2)
        # 合格证号
        cv2.rectangle(img,(top_left[0]+166,top_left[1]+294),(top_left[0]+403,top_left[1]+344),(255,0,0),2)
        # 发动机号
        cv2.rectangle(img,(top_left[0]+168,top_left[1]+346),(top_left[0]+535,top_left[1]+393),(255,0,0),2)
        # 车辆识别代码/车架号码
        cv2.rectangle(img, (top_left[0] + 752, top_left[1] + 345), (top_left[0] + 1094, top_left[1] + 395), (255,0,0), 2)
        # 价税合计
        cv2.rectangle(img, (top_left[0] + 898, top_left[1] + 394), (top_left[0] + 1049, top_left[1] + 443), (255,0,0), 2)

        #cv2.rectangle(img,top_left + 163)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()

if __name__ == "__main__":
    # adap_threshold()
    # ostu_binar()
    # geo_trans()
    # smooth_image()
    # morphological_tran()
    # image_gradients()
    # canny_edge()
    # img_pyramids()
    # contours()
    # histograms()
    temp_match()