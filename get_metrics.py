import ast
import argparse
import json
import os
from os.path import isfile, join

from utils.torch_utils import smart_inference_mode


def read_results_file(path, filename):
    out_dict = {}
    with open(os.path.join(path, filename), 'r') as file:
        for line in file:
            key = line.lstrip().split(':')[0]
            value = ast.literal_eval('{' + line.split('{')[1])
            out_dict[key] = value

    return out_dict


def write_output_file(save_path, filename, output_dict):
    with open(os.path.join(save_path, filename), 'w') as file:
        for key, value in output_dict.items():
            file.write(key + ": " + json.dumps(value) + '\n')


def count_metrics(in_dict):
    n_fp = 0
    n_tp = 0
    n_fn = 0
    n_tn = 0
    for value in in_dict.values():
        if 'TN' in value:
            n_tn = n_tn + value['TN']
        if 'TP' in value:
            n_tp = n_tp + value['TP']
        if 'FN' in value:
            n_fn = n_fn + value['FN']
        if 'FP' in value:
            n_fp = n_fp + value['FP']

    return n_fp, n_tp, n_fn, n_tn


@smart_inference_mode()
def run(
        path=None,  # Path where to find output file
        name=None,  # Name of output file to read
        save_path=None,  # Path where to save output file
        save_name=None  # Name of output file to save
):
    results = read_results_file(path, name)
    fp, tp, fn, tn = count_metrics(results)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn) # aka true positive rate

    tnr = tn / (tn + fp) # aka specificity
    fpr = 1 - tnr

    print("------------------------------------------------------------------")
    print("Results for " + name)
    print("FP: " + str(fp))
    print("TP: " + str(tp))
    print("FN: " + str(fn))
    print("TN: " + str(tn))
    print("Precision: " + str(precision * 100) + '%')
    print("True Positive Rate aka Recall: " + str(recall * 100) + '%')
    print("True Negative Rate aka Specificity: " + str(tnr * 100) + '%')
    print("False Positive Rate: " + str(fpr * 100) + '%')
    print("------------------------------------------------------------------")

    output_dict = {'FP': fp, 'TP': tp, 'FN': fn, 'TN': tn,
                   'Precision': precision, 'Recall': recall, 'TNR': tnr, 'FPR': fpr}

    write_output_file(save_path, save_name, output_dict)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path where to find output file')
    parser.add_argument('--name', type=str, help='Name of output file to read')
    parser.add_argument('--save-path', type=str, help='Path where to save output file')
    parser.add_argument('--save-name', type=str, help='Name of output file to save')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)