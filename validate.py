import argparse
import traceback
import pandas as pd
import numpy as np
import cv2 as cv
import os
from utils import get_precision, get_recall


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="List of image (txt)", type=str)
    parser.add_argument("--pred", help="Prediction result directory", type=str)
    return parser.parse_args()

def get_pred_result(img_list_file, pred_dir):
    with open(img_list_file, 'r') as file:
        img_name = file.readlines()
    img_path = [x.strip() for x in img_name] 
    filename = [os.path.basename(x.rsplit(".", 1)[0]) for x in img_path]
    filename = [os.path.join(pred_dir, x+".npy")for x in filename]
    return filename

def main(args):
    total_precision = np.zeros(99)
    total_recall = np.zeros(99)
    nums_precision = 0
    nums_recall = 0
    pred_result = get_pred_result(args.image, args.pred)
    
    for file in pred_result:
        gt_path = os.path.basename(file).split('.')[0] + '.txt'
        gt_txt = open(os.path.join("gt-txt/", gt_path))
        gt_coords = gt_txt.readlines()
        gt = [[
            int(float(l.rstrip().split(', ')[0])), 
            int(float(l.rstrip().split(', ')[1])), 
            int(float(l.rstrip().split(', ')[2])), 
            int(float(l.rstrip().split(', ')[3]))] for l in gt_coords]
        try:
            pred = np.load(file)
        except:
            continue

        for i in range(1, 99):
            p, num_p = get_precision(pred.tolist(), gt, threshold=i*0.01)
            r, num_r = get_recall(pred.tolist(), gt, threshold=i*0.01)
            total_precision[i-1] += p
            total_recall[i-1] += r
    
        nums_precision += num_p
        nums_recall += num_r


    total_recall = total_recall / nums_recall
    total_precision = total_precision / nums_precision
    f = 2 * np.divide((total_recall* total_precision), (total_recall + total_precision))

    print('Mean Precision:', total_precision.mean())
    print('Mean Recall:', total_recall.mean())
    print('Mean F:', f.mean()) 


if __name__ == "__main__":
    args = args_parser()
    try:
        main(args)
    except Exception as e:
        traceback.print_exc()