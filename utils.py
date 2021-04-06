import numpy as np 
import cv2 as cv
import pandas as pd
from metrics import EA_metric

def get_precision(pred, gt, threshold=0.90):
    N = len(pred)
    if N == 0:
        return 0, 0
    ea = np.zeros(N, dtype=np.float32)
    for i, coord_pred in enumerate(pred):
        if coord_pred[0]==coord_pred[2] and coord_pred[1]==coord_pred[3]:
            continue
        l_pred = coord_pred
        for coord_gt in gt:
            l_gt = coord_gt
            ea[i] = max(ea[i], EA_metric(l_pred, l_gt))
    return (ea >= threshold).sum(), N


def get_recall(pred, gt, threshold=0.90):
    N = len(gt)
    if N == 0:
        return 1.0, 0
    ea = np.zeros(N, dtype=np.float32)
    for i, coord_gt in enumerate(gt):
        l_gt = coord_gt
        for coord_pred in pred:
            if coord_pred[0]==coord_pred[2] and coord_pred[1]==coord_pred[3]:
                continue
            l_pred = coord_pred
            ea[i] = max(ea[i], EA_metric(l_pred, l_gt))
    return (ea >= threshold).sum(), N


def draw_matrix():
    pass