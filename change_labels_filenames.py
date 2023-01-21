import os
from os.path import isfile, join

labels_path = '/home/easousa/gitroot/datasets/FLIR/labels'

# label: FLIR_XXXXX_PreviewData.jpeg
# Change to just FLIR_XXXXX.jpg

labels_list = os.listdir(labels_path)

for label in labels_list:
    old_name = os.path.join(labels_path, label)
    new_name = os.path.join(labels_path, label.split('_')[0] + '_' + label.split('_')[1] + '.txt')
    os.rename(old_name, new_name)
