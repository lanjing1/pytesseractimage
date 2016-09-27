#!/usr/bin/python3
# -*- coding: utf8 -*-


import numpy as np
import cv2

# read show save image
'''
Second argument is a flag which specifies the way image should be read.
cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default
flag.
• cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
• cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
Note: Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively
'''


def img_operations():
    img = cv2.imread("src/image/drivinglicense1.png",0)
    cv2.imshow("image", img)
    k = cv2.waitKey(0)
    if k == 27:        # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite("drivinglicense1gray.png", img)
        cv2.destroyAllWindows()

# drawing different geometric shapes


def drawing_shapes():
    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.imshow("image", img)
    cv2.waitKey(0)

    # Draw a diagonal blue line with thickness of 5 px
    cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
    cv2.circle(img,(222,222),63,(0,0,255),-1)
    cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 90, 255, -1)
    # drawing polygon
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 255, 255))

    img = cv2.imread("blackboard.png")
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,"OpenCV",(10,500),font,4,(255,255,255),2)

    cv2.imshow("image", img)
    cv2.waitKey(0)

# Mouse as a Paint-Brush
def mouse_brush():
    pass


# Trackbar as the Color Palette
def trackbar_win():
    def nothing(x):
        pass
    # Create a black image, a window
    img = np.zeros((300,512,3),np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R','image',0,255,nothing)
    cv2.createTrackbar('G','image',0,255,nothing)
    cv2.createTrackbar('B','image',0,255,nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch,'image',0,1,nothing)
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R','image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch, 'image')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b,g,r]
    cv2.destroyAllWindows()
if __name__ == "__main__":
    print "Begining....."
    # drawing_shapes()
    trackbar_win()