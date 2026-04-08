# Copyright (c) 2020-2021 Nokia Corporation and/or its subsidiary(-ies). All rights reserved.
# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

#
# this script calcuates bpp for object detection and segmentation tasks

# modified by Sangwoon Kwak (s.kwak@etri.re.kr) and Joungil Yun (sigipus@etri.re.kr)

import pandas as pd
from PIL import Image
import glob
import os

QP_list=[22,27,32,37,42,47]


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default=None, \
    help='input image file directory')
parser.add_argument('--bitstream_dir', type=str, default=None, \
    help='encoded bitstream file director')
parser.add_argument('--input_fname', type=str, default=None, \
    help='input file that contains a list of image file names')
parser.add_argument('--output_data_file', type=str, default='bpp_data.csv', \
    help='output file')
parser.add_argument('--qp', type=int, default=22, \
    help='encoding qp (22, 27, 32, 37, 42, 47)')
parser.add_argument('--ds_level', type=int, default=0, \
    help='down-sacling level (0:100%, 1:75%, 2:50%, 3:25%)')


args = parser.parse_args()


input_fname = args.input_fname
output_data_file = args.output_data_file
bpp_data = []
bitstream_dir = args.bitstream_dir
input_dir = args.input_dir

qp = args.qp
ds_level = args.ds_level

##if img_list is None:
#file_list = sorted(glob.glob(os.path.join(input_dir, '*.*')))
#print(file_list)
#img_list = [file for file in file_list if
#                file.endswith('.png') or file.endswith('.jpg') or
#                file.endswith('.PNG') or file.endswith('.JPG')]

#print(img_list)    



input_list = None
if args.input_fname is not None:
    with open(args.input_fname, 'r') as f:
        input_list = [x for x in f.read().splitlines()] #remove .png
        
file_list_all = [(path, files) for path, dirs, files in os.walk(input_dir)]
file_list_all.sort()

file_list = {path: [file for file in files if
                    file.endswith('.png') or file.endswith('.jpg') or
                    file.endswith('.PNG') or file.endswith('.JPG')]
            for path, files in file_list_all}   



for key in file_list.keys():
    file_list[key].sort()

if input_list is not None:
    for file_path in file_list.keys():
        temp_list = file_list[file_path].copy()
        for file in temp_list:
            if file not in input_list:
                file_list[file_path].remove(file)
                

for file_path in file_list.keys():

    for idx, img_fname in enumerate(file_list[file_path]):
        img_id = os.path.splitext(os.path.basename(img_fname))[0]
        image_file_name = os.path.join(file_path, img_fname)
        img = Image.open(image_file_name)
        W,H = img.size
        #print('{} : {}'.format(idx, img_fname))
    
        bs_fname = os.path.join(bitstream_dir, f'{img_id}.bin')
        bs_size = os.path.getsize(bs_fname)
        bpp = bs_size * 8 / (W*H)
        bpp_data.append({ \
            'ImageID': img_id,
            'ImageWidth': W,
            'ImageHeight': H,
            'QP': qp,
            'DS' : ds_level,
            'BitstreamSize': bs_size,
            'Bpp': bpp })

bpp_data = pd.DataFrame(bpp_data)
bpp_data.to_csv(output_data_file, index=False)

scale = 100 - 25 * ds_level

# calculate the average in total bits and total pixels
total_bits = bpp_data['BitstreamSize'].sum() * 8
total_pixels = (bpp_data['ImageHeight'] * bpp_data['ImageWidth']).sum()
avg_bpp = total_bits / total_pixels

print('{} {:03d} {}'.format(qp, scale, avg_bpp))


