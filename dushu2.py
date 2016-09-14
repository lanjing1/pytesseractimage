#!/usr/bin/python3
# -*- coding: utf8 -*-
#######################
#
# 电表自动读数系统
#
#######################

import numpy as np
import imutils
import os

import pytesseract
from PIL import Image

import cv2

from skimage.morphology import disk
from skimage.filter import rank
import sys


from skimage import exposure
import argparse

show_img = True

def img_show_hook(title, img):
    global show_img
    type = sys.getfilesystemencoding()
    if show_img:
        #cv2.imshow(title, img)
        cv2.imshow(title.decode('utf-8').encode(type), img)
        cv2.waitKey(0)    
    return


def img_sobel_binary(im, blur_sz):
    
    # 高斯模糊，滤除多余的直角干扰
    img_blur = cv2.GaussianBlur(im,blur_sz,0)
    if len(img_blur.shape) == 3:
        # 转换成灰度图
        blur_gray = cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
    else:
        blur_gray = img_blur

    # 提取Sobel直角特征
    sobelx = cv2.Sobel(blur_gray, cv2.CV_16S, 1, 0, ksize=1)
    sobely = cv2.Sobel(blur_gray, cv2.CV_16S, 0, 1, ksize=1)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel_8ux = np.uint8(abs_sobelx)
    sobel_8uy = np.uint8(abs_sobely)
    # img_show_hook("Sobelx特征", sobel_8ux)
    # img_show_hook("Sobely特征", sobel_8uy)
    
    # OTSU提取二值图像    
    #ret, thdx = cv2.threshold(sobel_8ux, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret, thdy = cv2.threshold(sobel_8uy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, thdx = cv2.threshold(sobel_8ux, 12, 255, cv2.THRESH_BINARY)
    ret, thdy = cv2.threshold(sobel_8uy, 12, 255, cv2.THRESH_BINARY)

    thd_absx = cv2.convertScaleAbs(thdx)
    thd_absy = cv2.convertScaleAbs(thdy)
    bgimg = cv2.addWeighted(thd_absx, 0.5, thd_absy, 0.5, 0)
    
    img_show_hook("OTSU二值图像", bgimg)
    
    return bgimg


def img_contour_extra(im):
    # 腐蚀和膨胀
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # bgmask = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    bgmask = im
    img_show_hook("膨胀腐蚀结果", bgmask)
    
    # 获得连通区域
    # 该函数会破坏原始参数
    # findContours找到外部轮廓
    contours, hierarchy = cv2.findContours(bgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, contours, -1, (0, 0, 255), 3)

    cv2.imshow("img", im)
    cv2.waitKey(0)
    return contours


def img_contour_select(ctrs, im):
    # 剔除明显不满足条件的区域
    cand_rect = []
    for item in ctrs:
        # 周长，或者弧长，第二个参数表示该轮廓是否封闭，0.02的精度
        epsilon = 0.02*cv2.arcLength(item, True)
        # 进行轮廓近似
        approx = cv2.approxPolyDP(item, epsilon, True)  
        if len(approx) <= 8:
            # minAreaRect 获得这些轮廓的最小外接矩形(旋转的外包络矩形），存储在vector向量中，返回值是RotatedRect
            rect = cv2.minAreaRect(item)
            '''
                        if rect[1][0] < 20 or rect[1][1] < 20:
                continue
            if rect[1][0] > 150 or rect[1][1] > 150:
                continue
            '''

            # ratio = (rect[1][1]+0.00001) / rect[1][0]
            # if ratio > 1 or ratio < 0.9:
            #    continue
            # box = cv2.boxPoints(rect)
            # box是四个点的坐标
            box = cv2.cv.BoxPoints(rect)
            box_d = np.int0(box)
            #画出轮廓，-1,表示所有轮廓(0,表示画出第0个轮廓)，画笔颜色为(0, 255, 0)，即Green，粗细为3
            cv2.drawContours(im, [box_d], 0, (0,255,0), 3)
            cand_rect.append(box)
            img_show_hook("候选区域", im)
    #img_show_hook("候选区域", im)
    return cand_rect


# 轮廓
def img_contour_select_one(ctrs, im):
    # 剔除明显不满足条件的区域
    cand_rect = []
    for item in ctrs:
        # 周长，或者弧长，第二个参数表示该轮廓是否封闭，0.02的精度
        epsilon = 0.02*cv2.arcLength(item, True)
        # 进行轮廓近似
        approx = cv2.approxPolyDP(item, epsilon, True)
        if len(approx) == 4:
            # minAreaRect 获得这些轮廓的最小外接矩形(旋转的外包络矩形），存储在vector向量中，返回值是RotatedRect
            rect = cv2.minAreaRect(item)
            '''
                        if rect[1][0] < 20 or rect[1][1] < 20:
                continue
            if rect[1][0] > 150 or rect[1][1] > 150:
                continue
            '''

            # ratio = (rect[1][1]+0.00001) / rect[1][0]
            # if ratio > 1 or ratio < 0.9:
            #    continue
            # box = cv2.boxPoints(rect)
            # box是四个点的坐标
            box = cv2.cv.BoxPoints(rect)
            box_d = np.int0(box)
            #画出轮廓，-1,表示所有轮廓(0,表示画出第0个轮廓)，画笔颜色为(0, 255, 0)，即Green，粗细为3
            cv2.drawContours(im, [box_d], 0, (0,255,0), 2)
            cand_rect.append(box)
            img_show_hook("候选区域", im)
    #img_show_hook("候选区域", im)
    return cand_rect
def img_tesseract_detect(c_rect, im):
    # 由于使用minAreaRect获得的图像有-90~0的角度，所以给出的坐标顺序也不一定是
    # 转换时候给的，这里需要判断出图像的左上、左下、右上、右下的坐标，便于后面的变换
    pts = c_rect.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]
    
    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[2] = pts[np.argmin(diff)]
    rect[1] = pts[np.argmax(diff)]    

    dst = np.float32([[0,0],[0,100],[200,0],[200,100]])

    # 对于投影变换，我们则需要知道四个点，通过cv2.getPerspectiveTransform求得变换矩阵.
    # 之后使用cv2.warpPerspective获得矫正后的图片。
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(im, M, (200, 100))
    
    img_show_hook("剪裁识别图像", warp) 
    
    warp = np.array(warp, dtype=np.uint8)
    radius = 10
    selem = disk(radius)
    
    # 使用局部自适应OTSU阈值处理
    local_otsu = rank.otsu(warp, selem)
    l_otsu = np.uint8(warp >= local_otsu)
    l_otsu *= 255
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 4))
    # 膨胀和腐蚀操作的核函数
    l_otsu = cv2.morphologyEx(l_otsu, cv2.MORPH_CLOSE, kernel)
    
    img_show_hook("局部自适应OTSU图像", l_otsu) 
    
    print("识别结果：")
    print(pytesseract.image_to_string(Image.fromarray(l_otsu)))
    
    cv2.waitKey(0)
    return

# 检测直线
def img_hough_lines(im):
    im = cv2.GaussianBlur(im,(3,3),0)
    edges = cv2.Canny(im, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)  # 这里对最后一个参数使用了经验型的值
    result = im.copy()
    # 经验参数
    minLineLength = 200
    maxLineGap = 15
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)
    print(lines[0].min(axis=0))
    test = lines[0].min(axis=0)
    cv2.line(im, (test[0], test[1]), (test[2], test[3]), (0, 255, 0), 2)
    cv2.imshow('Cannyedgesone', im)


    for x1, y1, x2, y2 in lines[0]:
        #print(type(lines[0]))
        cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.imshow('Cannyedgesone', im)

    cv2.imshow('Cannyedges', edges)
    cv2.imshow('Result', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return im

# 找轮廓过滤矩形examples
def img_test(im):
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bgmask = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

    im = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)




    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)  # 这里对最后一个参数使用了经验型的值
    result = im.copy()
    # 经验参数
    minLineLength = 200
    maxLineGap = 15
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)


    for x1, y1, x2, y2 in lines[0]:
        cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Canny', edges)
    #edged = cv2.Canny(img_gray, 30, 200)



    #################

    # 图片二值化
    imagessss = img_sobel_binary(edges, (3, 3))
    ##################

    #cv2.imshow("edged", edged)
    cv2.waitKey(0)


    (cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:50]
    screenCnt = None
    # loop over our contours


    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            #break

            cv2.drawContours(img_gray, [screenCnt], -1, (0, 255, 0), 2)
            cv2.imshow("Game Boy Screen", img_gray)
            cv2.waitKey(0)


# 对图片进行裁剪、识别
def img_detect_tesseract(im):

    # 购买方名称及身份证号码/组织机构代码
    cv2.rectangle(im, (72, 80), (247, 112), (0, 255, 0), 2)
    cv2.imshow("购买方名称及身份证号码", im)

    # 创建图像
    #emptyImage = np.zeros((175,32), np.uint8)
    # 扣图像
    # box = (72, 80, 247, 112)
    # region = im.crop(box)
    # region.show()
    # 保存图像
    # cv2.imwrite("D:\\nameandid.jpg", img)

    # 发动机号码
    cv2.rectangle(im, (72, 162), (247, 185), (0, 255, 0), 2)
    cv2.imshow("发动机号码", im)

    # 车辆识别代码/车架号码
    cv2.rectangle(im, (353, 160), (518, 183), (0, 255, 0), 2)
    cv2.imshow("车辆识别代码/车架号码", im)

    # 价税合计（小写）
    cv2.rectangle(im, (425, 184), (466, 207), (0, 255, 0), 2)
    cv2.imshow("价税合计", im)

    print(pytesseract.image_to_string(Image.open('nameandid.png')))




if __name__ == "__main__":
    
    print("...图片文字识别系统...")
    
    #F1 = "172_79.jpg"
    #F1 = "633_88.jpg"
    #F1 = "22.png"
    #F1 = "reciept.jpg"
    #F1 = "reciept11.png"
    #F1 = "lion.png"
    F1 = "receiptrect.jpg"

    # 对图片进行裁剪、识别
    #img = Image.open(F1)


    img = cv2.imread(F1)
    img_show_hook("原图", img)
    # 改变图片的长宽比
    img = imutils.resize(img, width=516, height=331)
    img_detect_tesseract(img)

    #检测直线
    im = img_hough_lines(img)



    #img_show_hook("restest", img)


    #test 找轮廓找矩形
    img_test(img)

    # 转换成灰度图
    img_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # img_show_hook("灰度图像", img)
    # 得到二值图像
    sb_img = img_sobel_binary(im, (5,5))
    # img_show_hook("二值图像", sb_img)
    # 腐蚀和膨胀后，找到外部轮廓
    contours = img_contour_extra(sb_img)
    # 选出候选区域
    cand_rect = img_contour_select(contours, img)
    for item in cand_rect:
        # 输出识别结果
        img_tesseract_detect(np.array(item), img_gray)
    

# http://www.pyimagesearch.com/2014/03/10/building-pokedex-python-getting-started-step-1-6/
