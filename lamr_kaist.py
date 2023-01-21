import argparse
import ast
import math
import os
import platform
import sys
import json
from os.path import isfile, join
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

from utils.torch_utils import smart_inference_mode

kaist_txt_list = ['test-day-lwir-set6-v0.txt',
                  'test-day-lwir-set6-v1.txt',
                  'test-day-lwir-set6-v2.txt',
                  'test-day-lwir-set6-v3.txt',
                  'test-day-lwir-set6-v4.txt',
                  'test-day-lwir-set7-v0.txt',
                  'test-day-lwir-set7-v1.txt',
                  'test-day-lwir-set7-v2.txt',
                  'test-day-lwir-set8-v0.txt',
                  'test-day-lwir-set8-v1.txt',
                  'test-day-lwir-set8-v2.txt',
                  'test-night-lwir-set10-v0.txt',
                  'test-night-lwir-set10-v1.txt',
                  'test-night-lwir-set11-v0.txt',
                  'test-night-lwir-set11-v1.txt',
                  'test-night-lwir-set9-v0.txt',
                  'train-day-lwir-set0-v0.txt',
                  'train-day-lwir-set0-v1.txt',
                  'train-day-lwir-set0-v2.txt',
                  'train-day-lwir-set0-v3.txt',
                  'train-day-lwir-set0-v4.txt',
                  'train-day-lwir-set0-v5.txt',
                  'train-day-lwir-set0-v6.txt',
                  'train-day-lwir-set0-v7.txt',
                  'train-day-lwir-set0-v8.txt',
                  'train-day-lwir-set1-v0.txt',
                  'train-day-lwir-set1-v1.txt',
                  'train-day-lwir-set1-v2.txt',
                  'train-day-lwir-set1-v3.txt',
                  'train-day-lwir-set1-v4.txt',
                  'train-day-lwir-set1-v5.txt',
                  'train-day-lwir-set2-v0.txt',
                  'train-day-lwir-set2-v1.txt',
                  'train-day-lwir-set2-v2.txt',
                  'train-day-lwir-set2-v3.txt',
                  'train-day-lwir-set2-v4.txt',
                  'train-night-lwir-set3-v0.txt',
                  'train-night-lwir-set3-v1.txt',
                  'train-night-lwir-set4-v0.txt',
                  'train-night-lwir-set4-v1.txt',
                  'train-night-lwir-set5-v0.txt']


def find_nearest(in_list, value):
    return min(range(len(in_list)), key=lambda i: abs(in_list[i] - value))


def log_average(values):
    log_values = [math.log(value) for value in values]
    average_before_exp = sum(log_values) / len(values)
    return math.exp(average_before_exp)


def write_results_file(save_path, filename, output_dict):
    with open(os.path.join(save_path, filename), 'w') as file:
        for key, value in output_dict.items():
            file.write(key + ": " + json.dumps(value) + '\n')


def count_metrics(in_dict):
    n_tp_day = in_dict["Day"]["TP"]
    n_fp_day = in_dict["Day"]["FP"]
    n_fn_day = in_dict["Day"]["FN"]
    n_images_day = in_dict["Day"]["Number"]
    n_tp_night = in_dict["Night"]["TP"]
    n_fp_night = in_dict["Night"]["FP"]
    n_fn_night = in_dict["Night"]["FN"]
    n_images_night = in_dict["Night"]["Number"]

    return n_tp_day, n_fp_day, n_fn_day, n_images_day, n_tp_night, n_fp_night, n_fn_night, n_images_night


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


@smart_inference_mode()
def run(
        path=None,  # Path to folders containing detections at various thres levels
        save_path=None,  # Path to save output file to
        save_name=None,  # Name of output file
        output_table=None,  # Output LaTeX table
        var_thres=None  # Run this script for various iou_thres levels
):
    # In this file we apply the metrics described in "Pedestrian Detection: An Evaluation of the State of the Art"

    # In a nutshell:
    # Calculate the following metrics:
    # FPPI = FP / #images
    # Miss Rate = MR = FN / (TP + FN)
    # Change the values of C so that multiple values of FPPI and MR are generated and therefore be plotted

    # Then get Log Average Miss Rate (LAMR) "by averaging miss rate at nine
    # FPPI rates evenly spaced in log-space in the range 10^-22 to 10^0"

    # According to the same paper: For curves that end before reaching a given FPPI rate,
    # the minimum miss rate achieved is used)

    # Furthermore, according to "Eurocity Dataset" website:
    # In the absence of a miss-rate value for a given fppi, the highest existent fppi value is used.

    # I believe both mean the same. The highest FPPI will theoretically contain the lowest MR, which is the one used.
    # This matches what the paper means by "minimum MR achieved is used"

    # The 9 equally log spaced FPPI values are:
    # fppi_values = [0.01, 0.0178, 0.0316, 0.0562, 0.1, 0.1778, 0.3162, 0.5623, 1]
    fppi_values = np.logspace(-2, 0, 9).tolist()

    if not var_thres:
        fppi_list_day = []
        mr_list_day = []

        fppi_list_night = []
        mr_list_night = []

        fppi_list_total = []
        mr_list_total = []

        # Get filename lists
        dir_list = os.listdir(path)

        output_dict = {}

        for dir_name in dir_list:
            print(dir_name)
            results = read_results_file(os.path.join(path, dir_name), "lwir-var-thres.txt")

            n_tp_day, n_fp_day, n_fn_day, n_images_day, n_tp_night, n_fp_night, n_fn_night, n_images_night = count_metrics(
                results)

            # FPPI = FP / #images
            fppi_day = n_fp_day / n_images_day
            fppi_list_day.append(fppi_day)

            fppi_night = n_fp_night / n_images_night
            fppi_list_night.append(fppi_night)

            fppi_total = (n_fp_day + n_fp_night) / (n_images_day + n_images_night)
            fppi_list_total.append(fppi_total)

            # Miss Rate = MR = FN / (TP + FN)
            mr_day = n_fn_day / (n_tp_day + n_fn_day)
            mr_list_day.append(mr_day)

            mr_night = n_fn_night / (n_tp_night + n_fn_night)
            mr_list_night.append(mr_night)

            mr_total = (n_fn_day + n_fn_night) / ((n_tp_day + n_tp_night) + (n_fn_day + n_fn_night))
            mr_list_total.append(mr_total)
            # print("TP: " + str(n_tp) + "; FP: " + str(n_fp) + "; FN: " + str(n_fn))
            # print("FPPI: " + str(fppi) + "; MR: " + str(mr))
            # print('\n')
            day_dict = {"TP": n_tp_day, "FP:": n_fp_day, "FN": n_fn_day, "FPPI": fppi_day, "MR": mr_day}
            night_dict = {"TP": n_tp_night, "FP:": n_fp_night, "FN": n_fn_night, "FPPI": fppi_night, "MR": mr_night}
            total_dict = {"TP": n_tp_day + n_tp_night, "FP:": n_fp_day + n_fp_night, "FN": n_fn_day + n_fn_night,
                          "FPPI": fppi_total, "MR": mr_total}
            output_dict[dir_name] = {"Day": day_dict, "Night": night_dict, "Total": total_dict}

        mr_avg_list_day = []
        mr_avg_list_night = []
        mr_avg_list_total = []

        for i in range(0, 9):
            ref_fppi = fppi_values[i]
            fppi_index_day = find_nearest(fppi_list_day, ref_fppi)
            fppi_index_night = find_nearest(fppi_list_night, ref_fppi)
            fppi_index_total = find_nearest(fppi_list_total, ref_fppi)
            mr_avg_list_day.append(mr_list_day[fppi_index_day])
            mr_avg_list_night.append(mr_list_night[fppi_index_night])
            mr_avg_list_total.append(mr_list_total[fppi_index_total])

        lamr_day = log_average(mr_avg_list_day)
        lamr_night = log_average(mr_avg_list_night)
        lamr_total = log_average(mr_avg_list_total)
        lamr_dict = {"Day": lamr_day, "Night": lamr_night, "Total": lamr_total}

        output_dict["LAMR"] = lamr_dict

        write_results_file(save_path, save_name, output_dict)

        plot_lamr(save_path, fppi_list_day, mr_list_day, lamr_day, "day")
        plot_lamr(save_path, fppi_list_night, mr_list_night, lamr_night, "night")
        plot_lamr(save_path, fppi_list_total, mr_list_total, lamr_total, "total")

    else:

        iou_levels = [value / 100 for value in list(range(20, 100, 5))]

        output_dict = {}

        for iou_level in iou_levels:

            fppi_list_day = []
            mr_list_day = []

            fppi_list_night = []
            mr_list_night = []

            fppi_list_total = []
            mr_list_total = []

            # Get filename lists
            dir_list = os.listdir(path)

            out_dict = {}

            for dir_name in dir_list:
                print(dir_name)
                det_thres = float(dir_name.split("-")[3])
                filename = "lwir-var-thres_" + str(iou_level) + ".txt"
                results = read_results_file(os.path.join(path, dir_name), filename)

                n_tp_day, n_fp_day, n_fn_day, n_images_day, n_tp_night, n_fp_night, n_fn_night, n_images_night = count_metrics(
                    results)

                # FPPI = FP / #images
                fppi_day = n_fp_day / n_images_day
                fppi_list_day.append(fppi_day)

                fppi_night = n_fp_night / n_images_night
                fppi_list_night.append(fppi_night)

                fppi_total = (n_fp_day + n_fp_night) / (n_images_day + n_images_night)
                fppi_list_total.append(fppi_total)

                # Miss Rate = MR = FN / (TP + FN)
                mr_day = n_fn_day / (n_tp_day + n_fn_day)
                mr_list_day.append(mr_day)

                mr_night = n_fn_night / (n_tp_night + n_fn_night)
                mr_list_night.append(mr_night)

                mr_total = (n_fn_day + n_fn_night) / ((n_tp_day + n_tp_night) + (n_fn_day + n_fn_night))
                mr_list_total.append(mr_total)

                day_dict = {"TP": n_tp_day, "FP:": n_fp_day, "FN": n_fn_day, "FPPI": fppi_day, "MR": mr_day}
                night_dict = {"TP": n_tp_night, "FP:": n_fp_night, "FN": n_fn_night, "FPPI": fppi_night, "MR": mr_night}
                total_dict = {"TP": n_tp_day + n_tp_night, "FP:": n_fp_day + n_fp_night, "FN": n_fn_day + n_fn_night,
                              "FPPI": fppi_total, "MR": mr_total}
                time_of_day_dict = {"Day": day_dict, "Night": night_dict, "Total": total_dict}
                out_dict[det_thres] = time_of_day_dict

            mr_avg_list_day = []
            mr_avg_list_night = []
            mr_avg_list_total = []

            for i in range(0, 9):
                ref_fppi = fppi_values[i]
                fppi_index_day = find_nearest(fppi_list_day, ref_fppi)
                fppi_index_night = find_nearest(fppi_list_night, ref_fppi)
                fppi_index_total = find_nearest(fppi_list_total, ref_fppi)
                mr_avg_list_day.append(mr_list_day[fppi_index_day])
                mr_avg_list_night.append(mr_list_night[fppi_index_night])
                mr_avg_list_total.append(mr_list_total[fppi_index_total])

            lamr_day = log_average(mr_avg_list_day)
            lamr_night = log_average(mr_avg_list_night)
            lamr_total = log_average(mr_avg_list_total)
            lamr_dict = {"Day": lamr_day, "Night": lamr_night, "Total": lamr_total}

            out_dict["LAMR"] = lamr_dict
            output_dict[iou_level] = out_dict

            plot_lamr(save_path, fppi_list_day, mr_list_day, lamr_day, 'FPPI-MR-Plot_' + str(iou_level) + '_day.png')
            plot_lamr(save_path, fppi_list_night, mr_list_night, lamr_night, 'FPPI-MR-Plot_' + str(iou_level) + '_night.png')
            plot_lamr(save_path, fppi_list_total, mr_list_total, lamr_total, 'FPPI-MR-Plot_' + str(iou_level) + '_total.png')
            
        write_results_file(save_path, save_name, output_dict)


def plot_lamr(save_path, fppi_list, mr_list, lamr, save_name):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    fig.suptitle("LAMR = " + str(lamr))

    ax1.plot(fppi_list, mr_list, 'o-')
    ax1.set_xlabel('FPPI')
    ax1.set_ylabel('Miss Rate')
    ax1.grid(visible=True, which='both', axis='both', linestyle='--')

    ax2.loglog(fppi_list, mr_list, '.-')
    ax2.set_xlabel('FPPI')
    ax2.set_ylabel('Miss Rate')
    ax2.grid(visible=True, which='both', axis='both', linestyle='--')

    fig.savefig(os.path.join(save_path, save_name), bbox_inches='tight')
    #  fig.show()


def read_results_file(path, filename):
    out_dict = {}
    with open(os.path.join(path, filename), 'r') as file:
        for line in file:
            key = line.lstrip().split(':')[0]
            value = ast.literal_eval('{' + line.split('{')[1])
            out_dict[key] = value

    return out_dict


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to folders containing detections at various thres levels')
    parser.add_argument('--save-path', type=str, help='Path where to save output file')
    parser.add_argument('--save-name', type=str, help='Name of output file to save')
    parser.add_argument('--output-table', action='store_true', help='Output LaTeX table')
    parser.add_argument('--var-thres', action='store_true', help='Run this script for various iou_thres levels')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
