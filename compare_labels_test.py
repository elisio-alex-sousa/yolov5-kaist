import argparse
import math
import os
import platform
import sys
import json
from os.path import isfile, join
from pathlib import Path

import numpy as np

from utils.torch_utils import smart_inference_mode


# Inference
# results = model(im)

# results.pandas().xyxy[0]
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

# YOLO label format
# class - center_x - center_y - width - height - confidence (optional)

def convert_values(label1, label2, im_w, im_h):
    cx1 = float(label1.split(' ')[1])
    cy1 = float(label1.split(' ')[2])
    w1 = float(label1.split(' ')[3])
    h1 = float(label1.split(' ')[4])

    cx2 = float(label2.split(' ')[1])
    cy2 = float(label2.split(' ')[2])
    w2 = float(label2.split(' ')[3])
    h2 = float(label2.split(' ')[4])

    # Get absolute values from relative values
    abs_w1 = w1 * im_w
    abs_w2 = w2 * im_w
    abs_h1 = h1 * im_h
    abs_h2 = h2 * im_h
    abs_cx1 = cx1 * im_w
    abs_cx2 = cx2 * im_w
    abs_cy1 = cy1 * im_h
    abs_cy2 = cy2 * im_h

    # Get xmin
    xmin1 = abs_cx1 - abs_w1 / 2
    xmin2 = abs_cx2 - abs_w2 / 2

    # get xmax
    xmax1 = abs_cx1 + abs_w1 / 2
    xmax2 = abs_cx2 + abs_w2 / 2

    # get ymin
    ymin1 = abs_cy1 - abs_h1 / 2
    ymin2 = abs_cy2 - abs_h2 / 2

    # get ymax
    ymax1 = abs_cy1 + abs_h1 / 2
    ymax2 = abs_cy2 + abs_h2 / 2

    bb1 = {'x1': xmin1, 'x2': xmax1, 'y1': ymin1, 'y2': ymax1}
    bb2 = {'x1': xmin2, 'x2': xmax2, 'y1': ymin2, 'y2': ymax2}

    return bb1, bb2


def get_iou(bb1, bb2, name1, name2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}, aka {'xmin', 'xmax', 'ymin', 'ymax'}
        The (x1, y1) position is at the top left corner, aka xmin, ymin
        the (x2, y2) position is at the bottom right corner, aka xmax, ymax
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}, aka {'xmin', 'xmax', 'ymin', 'ymax'}
        The (x1, y1) position is at the top left corner, aka xmin, ymin
        the (x2, y2) position is at the bottom right corner, aka xmax, ymax

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    #print(name1 + " area: " + str(bb1_area))
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    #print(name2 + " area: " + str(bb2_area))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def main():

    pred_labels = ['0 0.0765625 0.474609 0.025 0.0507812 0.373833',
                   '0 0.713281 0.446289 0.0171875 0.0449219 0.396634',
                   '0 0.817969 0.446289 0.0234375 0.0605469 0.424763',
                   '0 0.916406 0.441406 0.0765625 0.113281 0.84403',
                   '0 0.301562 0.498047 0.0875 0.226562 0.847707',
                   '0 0.0265625 0.510742 0.05 0.216797 0.884303']

    truth_labels = ['0 0.29453125 0.4951171875 0.0953125 0.228515625',
                    '0 0.01875 0.5009765625 0.034375 0.212890625',
                    '0 0.10078125 0.462890625 0.0203125 0.0546875',
                    '0 0.078125 0.46484375 0.01875 0.0625',
                    '0 0.0625 0.4638671875 0.021875 0.068359375',
                    '0 0.16953125 0.4375 0.0203125 0.0390625',
                    '0 0.9171875 0.447265625 0.06875 0.109375']

    num_fp = 0
    num_tp = 0
    iou_thres = 0.45

    label_dict = {}
    ## Checking Areas, IoUs
    for i, pred_label in enumerate(pred_labels):
        iou_list = []
        for j, truth_label in enumerate(truth_labels):
            bb_pred, bb_truth = convert_values(pred_label, truth_label, 640, 512)
            iou = get_iou(bb_pred, bb_truth, i, j)
            iou_list.append(iou)
        max_index = np.argmax(iou_list)
        max_value = max(iou_list)

        if max_value > 0:
            label_dict[i] = max_index
        iou_list.clear()

    for i, pred_label in enumerate(pred_labels):
        for j, truth_label in enumerate(truth_labels):
            if i not in label_dict.keys():
                num_fp = num_fp + 1
                break
            else:
                if j == label_dict[i]:
                    bb_pred, bb_truth = convert_values(pred_label, truth_label, 640, 512)
                    iou = get_iou(bb_pred, bb_truth, i, j)
                    if iou < iou_thres:
                        num_fp = num_fp + 1
                    else:
                        num_tp = num_tp + 1

    num_fn = len(truth_labels) - len(pred_labels)

    print(label_dict)
    print("{'FP':" + str(num_fp) + ", 'TP':" + str(num_tp) + ", 'FN': " + str(num_fn) + "}")


if __name__ == "__main__":
    main()
