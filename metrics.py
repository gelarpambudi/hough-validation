'''
EVALUATION METRICS BASED ON DEEP HOUGH TRANSFORM
https://github.com/Hanqer/deep-hough-transform

Calculate Euclidan distance and Angular distance between two lines
'''

import numpy as np 
import cv2 as cv

def get_angle(pt1,pt2):
    if pt1[0] == pt2[0]:
        return -np.pi/2
    return np.arctan((pt1[1]-pt2[1])/(pt1[0]-pt2[0]))


def angular_dist(pred_coord, gt_coord):
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_coord[0], pred_coord[1], pred_coord[2], pred_coord[3]
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_coord[0], gt_coord[1], gt_coord[2], gt_coord[3]
    d_angle = np.abs(get_angle((pred_x1, pred_y1), (pred_x2, pred_y2)) - get_angle((gt_x1, gt_y1), (gt_x2, gt_y2)))
    d_angle = min(d_angle, np.pi-d_angle)
    d_angle = d_angle * 2 / np.pi
    return max(0, (1 - d_angle)) ** 2


def euclid_dist(pred_coord, gt_coord, size=(2000,2000)):
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_coord[0], pred_coord[1], pred_coord[2], pred_coord[3]
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_coord[0], gt_coord[1], gt_coord[2], gt_coord[3]
    c_p = [(pred_y1 + pred_y2) / 2, (pred_x1 + pred_x2) / 2]
    c_g = [(gt_y1+ gt_y2) / 2, (gt_x1 + gt_x2) / 2]
    d_coord = np.abs(c_p[0] - c_g[0])**2 + np.abs(c_p[1] - c_g[1])**2
    d_coord = np.sqrt(d_coord) / max(size[0], size[1])
    return max(0, (1 - d_coord)) ** 2


def EA_metric(pred_line, gt_line, size=(2000,2000)):
    angular = angular_dist(pred_line, gt_line)
    euclid = euclid_dist(pred_line, gt_line, size=size)
    return angular*euclid