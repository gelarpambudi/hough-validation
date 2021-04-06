import argparse
import traceback
import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from hough import get_powerline, hough_transform, convert_to_cartesian, pcnn

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="List of image (txt)", type=str)
    parser.add_argument("--hough", help="Hough algorithm (custom or original)", type=str)
    return parser.parse_args()


def predict_custom_hough(img_list_file):
    with open(img_list_file, 'r') as file:
        img_path = file.readlines()
    img_path = [x.strip() for x in img_path] 

    for img in tqdm(img_path):
        line_res = []
        pt1,pt2 = get_powerline(img)
        for i in range(len(pt1)):
            x1, y1 = pt1[i][0], pt1[i][1]
            x2, y2 = pt2[i][0], pt2[i][1]  
            line_res.append((x1, y1, x2, y2))

        np_line = np.array(line_res)
        filename = os.path.basename(img.rsplit(".", 1)[0])
        np.save(os.path.join("pred-result-custom/",filename),np_line)


def predict_orig_hough(img_list_file):
    with open(img_list_file, 'r') as file:
        img_path = file.readlines()
    img_path = [x.strip() for x in img_path]

    for img in tqdm(img_path):
        line_res = []
        input_image = cv.imread(img)
        hough_res = hough_transform(input_image,350)
        if hough_res is None:
            pass
        else:
            pt1, pt2 = convert_to_cartesian(hough_res)
            for i in range(len(pt1)):
                x1, y1 = pt1[i][0], pt1[i][1]
                x2, y2 = pt2[i][0], pt2[i][1]  
                line_res.append((x1, y1, x2, y2))

        np_line = np.array(line_res)
        filename = os.path.basename(img.rsplit(".", 1)[0])
        np.save(os.path.join("pred-result-ori/",filename),np_line)


if __name__ == "__main__":
    args = args_parser()
    if args.hough == "custom":
        try:
            predict_custom_hough(args.image)
        except Exception as e:
            traceback.print_exc()
    elif args.hough == "original":
        try:
            predict_orig_hough(args.image)
        except Exception as e:
            traceback.print_exc()