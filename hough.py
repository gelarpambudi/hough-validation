import numpy as np
import cv2 as cv
from scipy import signal
import math
import pandas as pd

Alpha_F = 0.1
Alpha_L = 1.0
Alpha_T = 0.3
V_F = 0.5
V_L = 0.2
V_T = 20.0
Num = 10
Beta = 0.1
W = [0.5, 1, 0.5, 1, 0, 1, 0.5, 1, 0.5,]
W = np.array(W, np.float).reshape((3, 3))
M = W
global F
global L
global Y
global T
global Y_AC

def pcnn(input_image):
    src = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    dim = src.shape

    F = np.zeros( dim, np.float)
    L = np.zeros( dim, np.float)
    Y = np.zeros( dim, np.float)
    T = np.ones( dim, np.float)
    Y_AC = np.zeros( dim, np.float)
    S = cv.normalize(src.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    
    for cont in range(Num):
        F = np.exp(-Alpha_F) * F + V_F * signal.convolve2d(Y, W, mode='same') + S
        L = np.exp(-Alpha_L) * L + V_L * signal.convolve2d(Y, M, mode='same')
        U = F * (1 + Beta * L)
        T = np.exp(-Alpha_T) * T + V_T * Y
        Y = (U>T).astype(np.float)
        Y_AC = Y_AC + Y
    Y_AC = cv.normalize(Y_AC.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    return Y_AC


def hough_transform(input_image, hough_thres):
    Y_AC = pcnn(input_image)
    edges = cv.Canny((Y_AC*255).astype(np.uint8),100,100,apertureSize = 3)
    lines_pcnn = cv.HoughLines(edges,1,np.pi/180,hough_thres)
    return lines_pcnn


def get_powerline(input_image):
    img = cv.imread(input_image)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    num_line = 10000

    for kernel_size in range(3,15,2):
        img = cv.bilateralFilter(img, 3, 250, 250)
        i = 100
        while (i <= 400):    
            lines = hough_transform(img, i)
            if lines is None:
                break
            if (lines.shape[0] < num_line) and (lines.shape[0] >= 2):
                num_line = lines.shape[0]
                result = lines
            i = i + 20
        if (lines is None) and (result.shape[0] <= 3) and (result.shape[0] >= 1):
            break
    
    pt1,pt2 = convert_to_cartesian(result)
    return pt1,pt2
    

def convert_to_cartesian(hough_res, img_width=2000, img_height=2000, xmin=0, ymin=0):
    pt1_list = []
    pt2_list = []
    for i in range(0, len(hough_res)):
        rho = hough_res[i][0][0]
        theta = hough_res[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = [int(x0 + img_width*(-b) + xmin), int(y0 + img_height*(a)) + ymin]
        pt2 = [int(x0 - img_width*(-b) + xmin), int(y0 - img_height*(a)) + ymin]
        pt1_list.append(pt1)
        pt2_list.append(pt2)
    return pt1_list, pt2_list


