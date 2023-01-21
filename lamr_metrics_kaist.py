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


def write_results_file(save_path, filename, output_dict):
    with open(os.path.join(save_path, filename), 'w') as file:
        for key, value in output_dict.items():
            file.write(key + ": " + json.dumps(value) + '\n')


day_sets = [0, 1, 2, 6, 7, 8]
night_sets = [3, 4, 5, 9, 10, 11]
test_sets = list(range(6, 12))
train_sets = list(range(0, 6))
set0_v = list(range(0, 9))
set1_v = list(range(0, 6))
set2_v = list(range(0, 5))
set3_v = list(range(0, 2))
set4_v = list(range(0, 2))
set5_v = [0]
set6_v = list(range(0, 5))
set7_v = list(range(0, 3))
set8_v = list(range(0, 3))
set9_v = [0]
set10_v = list(range(0, 2))
set11_v = list(range(0, 2))
sets = {0: set0_v, 1: set1_v, 2: set2_v, 3: set3_v, 4: set4_v, 5: set5_v, 6: set6_v, 7: set7_v, 8: set8_v, 9: set9_v,
        10: set10_v, 11: set11_v}


def build_kaist_subdir(s, v):
    if s < 10:
        set_subdir = "/set0" + str(s) + "/V00" + str(v) + "/lwir/"
    else:
        set_subdir = "/set" + str(s) + "/V00" + str(v) + "/lwir/"

    return set_subdir


def build_script_subdir(s, v, level):
    pred_subdir = ""

    if s in test_sets:
        pred_subdir = pred_subdir + "test-"
    else:
        pred_subdir = pred_subdir + "train-"

    if s in day_sets:
        pred_subdir = pred_subdir + "day-lwir-"
    else:
        pred_subdir = pred_subdir + "night-lwir-"

    pred_subdir = pred_subdir + "set" + str(s) + "-v" + str(v) + "-thres-" + str(level) + "/labels/"

    return pred_subdir


def get_files_list(path_pred, path_truth, path_images, s, v, level):
    set_subdir = build_kaist_subdir(s, v)
    image_list = path_images + set_subdir
    truth_list = path_truth + set_subdir

    pred_subdir = build_script_subdir(s, v, level)

    pred_list = os.path.join(path_pred, pred_subdir)

    return pred_list, truth_list, image_list

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


def get_iou(bb1, bb2):
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
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def produce_metrics(height, img_list, iou_thres, pred_dict, truth_dict, width):
    # In this file we apply the metrics described in "Pedestrian Detection: An Evaluation of the State of the Art"

    # In a nutshell:

    # - Let IoU threshold be 'C'
    # - IoU must be > C in order to be a hit (True Positive) (Leave at 0.5 by default)
    # - Unmatched BBdt counts as FP (i.e. a label in pred not matched to a label in truth)
    # - Unmatched BBgt counts as FN (i.e. a label in truth not matched to a label in pred)
    # (TN are not considered here, since we are not detecting negative cases)

    results = {}

    sum_tp = 0
    sum_fp = 0
    sum_fn = 0

    # Go through each image
    # print('\n')
    for ix, img in enumerate(img_list):
        # print('Image ' + img + ' [' + str(ix + 1) + '/' + str(len(img_list)) + ']', end='\r')

        # Get name of img without extension
        img_name = os.path.splitext(img)[0]

        results[img_name] = {}

        # False Negatives (pred is empty, truth is not)
        if img_name not in pred_dict and img_name in truth_dict:
            results[img_name] = {'FN': len(truth_dict[img_name])}
            sum_fn = sum_fn + 1
        # False Positives (truth is empty, pred is not)
        if img_name not in truth_dict and img_name in pred_dict:
            results[img_name] = {'FP': len(pred_dict[img_name])}
            sum_fp = sum_fp + 1
        # If both are not empty:
        if img_name in truth_dict and img_name in pred_dict:
            # Get labels
            truth_labels = truth_dict[img_name]
            pred_labels = pred_dict[img_name]

            label_dict = {}
            # This for cycle fills in "label_dict" and associates each pred label to its matching truth label
            # according to their IoU levels
            # So for example label_dict[0] = 1 means that first pred label is matched to second truth label
            for i, pred_label in enumerate(pred_labels):
                iou_list = []
                for j, truth_label in enumerate(truth_labels):
                    bb_pred, bb_truth = convert_values(pred_label, truth_label, width, height)
                    iou = get_iou(bb_pred, bb_truth)
                    iou_list.append(iou if iou > iou_thres else 0)  # Added to only add correct overlaps
                max_index = np.argmax(iou_list)
                max_value = max(iou_list)

                if max_value > 0:
                    label_dict[i] = max_index
                iou_list.clear()

            num_fp = 0
            num_fn = 0
            num_tp = 0
            # For each pred and truth label:
            for i, pred_label in enumerate(pred_labels):
                for j, truth_label in enumerate(truth_labels):
                    # If the pred label does not have a match with a truth label, it is a False Positive
                    if i not in label_dict.keys():
                        num_fp = num_fp + 1
                    # If the pred label is in the truth label, then there was a detection with IoU > thres
                    # then it is a TP
                    else:
                        if j == label_dict[i]:
                            num_tp = num_tp + 1

            # For every truth label not matched to a pred label, count as missing detection
            for j, truth_label in enumerate(truth_labels):
                if j not in label_dict.values():
                    num_fn = num_fn + 1

            results[img_name] = {}

            if num_fp > 0:
                results[img_name].update({'FP': num_fp})
                sum_fp = sum_fp + num_fp

            if num_fn > 0:
                results[img_name].update({'FN': num_fn})
                sum_fn = sum_fn + num_fn

            if num_tp > 0:
                results[img_name].update({'TP': num_tp})
                sum_tp = sum_tp + num_tp

    return results, sum_tp, sum_fp, sum_fn


@smart_inference_mode()
def run(
        name=None,  # Name of output file
        save_path=None,  # Path to save output file to
        path_pred=None,  # Path to labeled predictions
        path_truth=None,  # Path to labeled truths
        path_images=None,  # Path to JPEG images
        iou_thres=0.45,  # Acceptance threshold for IoU values
        thres_level=None,  # Detection threshold
        var_thres=None,  # Run this script for various iou_thres levels
        width=640,  # Image width
        height=512  # Image height
):
    if not var_thres:

        n_imgs_day, n_imgs_night, fn_day, fn_night, fp_day, fp_night, tp_day, tp_night = \
            get_metrics_by_time_of_day(height, iou_thres, path_images, path_pred, path_truth, thres_level, width)

        results = {"Day": {"TP": tp_day, "FP": fp_day, "FN": fn_day, "Number": n_imgs_day},
                   "Night": {"TP": tp_night, "FP": fp_night, "FN": fn_night, "Number": n_imgs_night}}

        write_results_file(save_path, name, results)

    else:

        results = {}
        iou_levels = [value / 100 for value in list(range(20, 100, 5))]

        for iou_level in iou_levels:
            print("IOU: " + str(iou_level))
            n_imgs_day, n_imgs_night, fn_day, fn_night, fp_day, fp_night, tp_day, tp_night = \
                get_metrics_by_time_of_day(height, iou_level, path_images, path_pred, path_truth, thres_level, width)

            current_result = {"Day": {"TP": tp_day, "FP": fp_day, "FN": fn_day, "Number": n_imgs_day},
                              "Night": {"TP": tp_night, "FP": fp_night, "FN": fn_night, "Number": n_imgs_night}}

            results.update({iou_level: current_result})

            save_name = name.split(".")[0] + "_" + str(iou_levels[0]) + "_" + name.split(".")[1]
            write_results_file(save_path, save_name, results)


def get_metrics_by_time_of_day(height, iou_thres, path_images, path_pred, path_truth, thres_level, width):
    sum_tp_day = 0
    sum_tp_night = 0
    sum_fp_day = 0
    sum_fp_night = 0
    sum_fn_day = 0
    sum_fn_night = 0
    number_of_images_day = 0
    number_of_images_night = 0
    for s_number in sets.keys():
        for v_number in sets[s_number]:

            print("S: " + str(s_number) + ", V: " + str(v_number))

            path_pred_full, path_truth_full, path_images_full = get_files_list(path_pred, path_truth, path_images,
                                                                               s_number, v_number, thres_level)
            # Get filename lists
            pred_list = [f for f in os.listdir(path_pred_full) if isfile(join(path_pred_full, f))]
            truth_list = [f for f in os.listdir(path_truth_full) if isfile(join(path_truth_full, f))]
            img_list = [f for f in os.listdir(path_images_full) if isfile(join(path_images_full, f))]

            # print("Path pred: " + path_pred_full)
            # print("Path truth: " + path_truth_full)
            # print("Path image: " + path_images_full)

            if s_number in day_sets:
                number_of_images_day = number_of_images_day + len(img_list)

            if s_number in night_sets:
                number_of_images_night = number_of_images_night + len(img_list)

            pred_dict = {}
            truth_dict = {}

            # Get lines for each label list
            for filename in pred_list:
                with open(join(path_pred_full, filename)) as file:
                    pred_dict[os.path.splitext(filename)[0]] = [line.rstrip() for line in file]

            pred_dict = {k: v for k, v in pred_dict.items() if
                         v}  # Remove empty labels; images where no pedestrians detected

            for filename in truth_list:
                with open(join(path_truth_full, filename)) as file:
                    truth_dict[os.path.splitext(filename)[0]] = [line.rstrip() for line in file]

            truth_dict = {k: v for k, v in truth_dict.items() if v}  # Remove empty labels

            results, tp, fp, fn = produce_metrics(height, img_list, iou_thres, pred_dict, truth_dict, width)

            if s_number in day_sets:
                sum_tp_day = sum_tp_day + tp
                sum_fp_day = sum_fp_day + fp
                sum_fn_day = sum_fn_day + fn

            if s_number in night_sets:
                sum_tp_night = sum_tp_night + tp
                sum_fp_night = sum_fp_night + fp
                sum_fn_night = sum_fn_night + fn
    return number_of_images_day, number_of_images_night, sum_fn_day, sum_fn_night, sum_fp_day, sum_fp_night, sum_tp_day, sum_tp_night


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, help='Path where to save output file')
    parser.add_argument('--name', type=str, help='Name of output file')
    parser.add_argument('--path-pred', type=str, help='Path to TXT files of labeled predictions')
    parser.add_argument('--path-truth', type=str, help='Path to TXT files of labeled truths')
    parser.add_argument('--path-images', type=str, help='Path to JPEG files of images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='Acceptance threshold for IoU values')
    parser.add_argument('--thres-level', type=float, help="Detection threshold")
    parser.add_argument('--var-thres', action='store_true', help='Run this script for various iou_thres levels')
    parser.add_argument('--width', type=int, default=640, help="Image width")
    parser.add_argument('--height', type=int, default=512, help="Image height")
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
