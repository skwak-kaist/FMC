# Copyright (c) 2020-2021 Nokia Corporation and/or its subsidiary(-ies). All rights reserved.
# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

#
# This script generates segmentation output file from a detection result

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import oid_mask_encoding

parser = argparse.ArgumentParser()
parser.add_argument('--input_annotations', type=str, \
  help='segmentation groundtruth')
parser.add_argument('--gt_mask_dir', type=str, \
  help='groundtruth mask directory'),
parser.add_argument('--input_predictions', type=str, \
  help='prediction output file in OpenImages format')
parser.add_argument('--output_annotations', type=str, \
  help='output segmentation output that the mask are resized to match the predictions')

args = parser.parse_args()

anno_gt = pd.read_csv(args.input_annotations)
pred = pd.read_csv(args.input_predictions)

for idx, row in anno_gt.iterrows():
  pred_rslt = pred.loc[pred['ImageID']==row['ImageID']]
  if len(pred_rslt)==0:
    print('Image not in prediction: ', row['ImageID'])
    continue

  W = pred_rslt['ImageWidth'].iloc[0]
  H = pred_rslt['ImageHeight'].iloc[0]

  

  # resize mask if necessary
  mask_img = Image.open(os.path.join(args.gt_mask_dir, row['MaskPath']))

  if (mask_img.size[0]!=W) or (mask_img.size[1]!=H):
    mask_img = mask_img.resize((W,H))
    mask = np.asarray(mask_img)
    mask_str = oid_mask_encoding.encode_binary_mask(mask).decode('ascii')
    anno_gt.at[idx, 'Mask'] = mask_str
    anno_gt.at[idx, 'ImageWidth'] = W
    anno_gt.at[idx, 'ImageHeight'] = H

anno_gt.to_csv(args.output_annotations, index=False)





