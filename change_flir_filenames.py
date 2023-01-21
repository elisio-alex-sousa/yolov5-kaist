import os
from os.path import isfile, join

imgs_path = '/home/easousa/gitroot/datasets/FLIR/JPEGImages'

lwir_folder = 'lwir'
rgb_folder = 'visible'

lwir_path = os.path.join(imgs_path, lwir_folder)
rgb_path = os.path.join(imgs_path, rgb_folder)

# lwir img: FLIR_XXXXX_PreviewData.jpeg
# visible img: FLIR_XXXXX_RGB.jpg
# Change both to just FLIR_XXXXX.jpg

lwir_img_list = os.listdir(lwir_path)
rgb_img_list = os.listdir(rgb_path)

for img in lwir_img_list:
    old_name = os.path.join(lwir_path, img)
    new_name = os.path.join(lwir_path, img.split('_')[0] + '_' + img.split('_')[1] + '.jpeg')
    os.rename(old_name, new_name)

for img in rgb_img_list:
    old_name = os.path.join(rgb_path, img)
    new_name = os.path.join(rgb_path, img.split('_')[0] + '_' + img.split('_')[1] + '.jpeg')
    os.rename(old_name, new_name)
