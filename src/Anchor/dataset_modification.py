##
## Dataset Modification Test code for VCM CfP idea
##
## by Sangwoon Kwak (s.kwak@etri.re.kr) and Joungil Yun (sigipus@etri.re.kr)
## 

import torch, torchvision
import detectron2

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, cv2, random
import matplotlib.pyplot as plt
import argparse

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import oid_mask_encoding

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from skimage import io
from torch import nn
from torch.nn import functional as F
from PIL import Image

def create_path(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("[Error] Cannot create a path: {}".format(path))


def display_multi_images(images, rows = 1, cols=1, color=None):
    figure, ax = plt.subplots(nrows=rows, ncols=cols)
    for ind,title in enumerate(images):
        ax.ravel()[ind].imshow(images[title], cmap=color)
        ax.ravel()[ind].set_title(title)
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()

    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()
    plt.show()


class ForwardBackwardHook(object):
    def __init__(self, model):
        self.model = model
        
        self._modules = []
        self._handlers = []

        self.features = []
        self.grads = []
                
    def remove(self):
        for handler in self._handlers:
            handler.remove()
        self._handlers.clear()
        self._modules.clear()

    def clear(self):
        self.features.clear()
        self.grads.clear()
        
    def _forward_func(self, module, feature_input, feature_output):
        if isinstance(feature_output, list): # for top_block
            self.features.append(feature_output[0])
        else:
            self.features.append(feature_output)

    def _backward_func(self, module, grad_input, grad_output):
        self.grads.insert(0, grad_output[0])
        
    def register(self, stem_flag=True, c2_flag=False, p_flag=False, last_flag=False):
        self.remove()

        if stem_flag:
            self._modules.append(self.model.backbone.bottom_up.stem)
        
        if c2_flag:
            self._modules.append(self.model.backbone.bottom_up.res2)
        
        if p_flag:
            self._modules.append(self.model.backbone.fpn_output2)
            self._modules.append(self.model.backbone.fpn_output3)
            self._modules.append(self.model.backbone.fpn_output4)
            self._modules.append(self.model.backbone.fpn_output5)
            self._modules.append(self.model.backbone.top_block)

        for module in self._modules:
            self._handlers.append(module.register_forward_hook(self._forward_func))
            self._handlers.append(module.register_backward_hook(self._backward_func))



parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./', \
    help='input image file directory')
parser.add_argument('--modified_dir', type=str, default=None, \
    help='modified output directory')
parser.add_argument('--task', type=str, default='detection', \
    help='task: detection or segmentation')
parser.add_argument('--config_file', type=str, default='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', \
    help='model config file for detectron2')
parser.add_argument('--input_file', type=str, default=None, \
    help='input file that contains a list of image file names')

parser.add_argument('--block_size', type=int, default=32, \
    help='block size for the block-wise degradation')
parser.add_argument('--num_levles', type=int, default=4, \
    help='number of image levels')
parser.add_argument('--degrade_level', type=float, default=0.25, \
    help='minimum degradation level')
parser.add_argument('--alpha', type=float, default=0.5, \
    help='Ratio of the original block')
parser.add_argument('--visualize_flag', type=bool, default=False, \
    help='Visualize the plots')
parser.add_argument('--filter_type', type=int, default=0, \
    help='Filter type for interpolation')
parser.add_argument('--loss_type', type=int, default=2, \
help='feature loss type, int value 1: L1loss, 2:L2loss')
parser.add_argument('--num_QP', type=int, default=4, \
help='list of input QP')
parser.add_argument('--dqp', type=int, default=0, \
help='the value of dQP')

args = parser.parse_args()

if args.task == 'detection':
  model_cfg_name = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
elif args.task == 'segmentation':
  model_cfg_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
else:
  assert False, print("Unrecognized task:", args.task)

# construct detectron2 model
print('constructing detectron model ...')


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(model_cfg_name))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg_name)
predictor = DefaultPredictor(cfg)


if args.modified_dir is None:
    assert False, print("Modified output directory should be given")

# s.kwak Params
block_size = args.block_size
degrade_level = args.degrade_level
alpha = args.alpha
loss_type=args.loss_type

# s.kwak Flags
visualize_flag = args.visualize_flag
QP_List = [22, 27, 32, 37, 42, 47]


modified_dir = args.modified_dir
create_path(modified_dir)
create_path(modified_dir + '_binarymap')
create_path(modified_dir + '_QPmap')
create_path(modified_dir + '_degraded')

for qp in QP_List[:args.num_QP]:
    create_path(modified_dir + '_QPmap/' + str(qp))

layer_selection_flags = [False, False, True]  # [stem, c2, p]

model_0 = build_model(cfg)
checkpointer_0 = DetectionCheckpointer(model_0)
checkpointer_0.load(cfg.MODEL.WEIGHTS)
hook_0 = ForwardBackwardHook(model_0)
hook_0.register(*layer_selection_flags)


model_1 = build_model(cfg)
checkpointer_1 = DetectionCheckpointer(model_1)
checkpointer_1.load(cfg.MODEL.WEIGHTS)
hook_1 = ForwardBackwardHook(model_1)
hook_1.register(*layer_selection_flags)

model_0.eval()
model_1.eval()  



with open(args.input_file, 'r') as f:
    for i, img_fname in enumerate(f.readlines()):
        
        img_fname = os.path.join(args.input_dir, img_fname.strip())
        original_image = cv2.imread(img_fname)

        assert original_image is not None, print(f'Image file not found: {img_fname}')
       
        print(f'processing {img_fname}...')

        # resolution of original image
        height, width = original_image.shape[:2]
        width_num = int(np.ceil(width / block_size))
        height_num = int(np.ceil(height /block_size))

        # Generate degraded images
        degraded_image = cv2.resize(original_image, dsize=(0,0), fx=degrade_level, fy=degrade_level, interpolation=cv2.INTER_CUBIC)
        if args.filter_type==0: # bicubic
            degraded_image = cv2.resize(degraded_image, dsize=(width,height), interpolation=cv2.INTER_CUBIC)
        elif args.filter_type==1: # gaussian blur
            degraded_image = cv2.GaussianBlur(original_image, (7, 7), 0)
            degraded_image = cv2.GaussianBlur(degraded_image, (5, 5), 0)
            degraded_image = cv2.GaussianBlur(degraded_image, (5, 5), 0)
            #degraded_image = cv2.GaussianBlur(degraded_image, (5, 5), 0)
            #degraded_image = cv2.GaussianBlur(degraded_image, (5, 5), 0)

        elif args.filter_type==2: # nearest neighbor
            degraded_image = cv2.resize(degraded_image, dsize=(width,height), interpolation=cv2.INTER_NEAREST)
        elif args.filter_type==3: # bilateral filter
            degraded_image = cv2.bilateralFilter(original_image, -1, 10, 5)
            degraded_image = cv2.bilateralFilter(degraded_image, -1, 10, 5) 
            degraded_image = cv2.bilateralFilter(degraded_image, -1, 10, 5)
        else:
            degraded_image = cv2.resize(degraded_image, dsize=(width,height), interpolation=cv2.INTER_CUBIC)

        # Calculate feature loss between two images
        model_0.zero_grad()
        model_1.zero_grad()

        hook_0.clear()
        hook_1.clear()

        image_0 = original_image
        image_0 = torch.as_tensor(image_0.astype("float32").transpose(2, 0, 1)).requires_grad_(True)
        inputs_0 = {"image": image_0, "height": height, "width": width}
        output_0 = model_0.inference([inputs_0])[0]['instances']

        image_1 = degraded_image
        image_1 = torch.as_tensor(image_1.astype("float32").transpose(2, 0, 1)).requires_grad_(True)
        inputs_1 = {"image": image_1, "height": height, "width": width}    
        output_1 = model_1.inference([inputs_1])[0]['instances']       
             
        for i, (f0, f1) in enumerate(zip(hook_0.features, hook_1.features)):
            if i == 0:
                feature_loss = torch.sum(torch.norm(f0-f1, loss_type))
            else:
                feature_loss += torch.sum(torch.norm(f0-f1, loss_type))
        feature_loss.backward()


        # Find absolute gradient map
        grad_cat = torch.cat([image_0.grad.data, image_1.grad.data], dim=0)
        grad_image = grad_cat.cpu().numpy().transpose(1,2,0)
        grad_image = np.abs(grad_image)
        grad_image = np.max(grad_image, axis=2)

        #print(np.max(grad_image))
        #print(np.min(grad_image))
        #print(np.mean(grad_image))

        # Called explicitly due to GPU usage memory increase issue
        del image_0
        del image_1
        del output_0
        del output_1
        del grad_cat
        torch.cuda.empty_cache()


        # Find the representative value for each block
        block_value_map = np.zeros((height_num, width_num), dtype=float)
        for h in range(height_num):
            h0 = h * block_size
            h1 = np.minimum((h+1) * block_size, height)

            for w in range(width_num):
                w0 = w * block_size
                w1 = np.minimum((w+1) * block_size, width)

                max_win = np.max(grad_image[h0:h1, w0:w1])
                avg_win = np.mean(grad_image[h0:h1, w0:w1])

                block_value_map[h,w] = avg_win


        # Find threshold value for top N blocks
        sorted_vector = np.sort(block_value_map.flatten())
        selected_th_value = sorted_vector[int((len(sorted_vector)-1) * (1-alpha))]

        # generate modifed image
        modified_image = np.zeros(original_image.shape).astype('uint8')
        binary_map = np.zeros((height_num, width_num))
        QP_map_base = np.zeros((height, width))
        for h in range(height_num):
            h0 = h * block_size
            h1 = np.minimum((h+1) * block_size, height)

            for w in range(width_num):
                w0 = w * block_size
                w1 = np.minimum((w+1) * block_size, width)

                if block_value_map[h,w] >= selected_th_value:
                    modified_image[h0:h1, w0:w1, :] = original_image[h0:h1, w0:w1, :]
                    QP_map_base[h0:h1, w0:w1] = args.dqp
                    binary_map[h,w]=1
                else: 
                    modified_image[h0:h1, w0:w1, :] = degraded_image[h0:h1, w0:w1, :]
                    binary_map[h,w]=0

        # visualization
        if visualize_flag:                      
            # plt.imshow(block_value_map/np.max(block_value_map), cmap='gray', vmin=0, vmax=1.0)
            # plt.show() 
            block_value_map_gray = block_value_map/np.max(block_value_map)
            display_multi_images({"original": original_image, "degraded": degraded_image, "modified": modified_image,  \
                "gradient activation": grad_image, "gradient activation(block)": block_value_map, "Binary map": binary_map}, 2, 3)

        modified_file = os.path.join(modified_dir, os.path.splitext(os.path.basename(img_fname))[0] + '.png')
        cv2.imwrite(modified_file, modified_image)       

        degraded_file = os.path.join(modified_dir + '_degraded', os.path.splitext(os.path.basename(img_fname))[0] + '.png')
        cv2.imwrite(degraded_file, degraded_image)       
               
        quality_map_file = os.path.join(modified_dir + '_binarymap', os.path.splitext(os.path.basename(img_fname))[0] + '.txt')
        np.savetxt(quality_map_file, binary_map, fmt='%d', delimiter=' ')      
        
        if args.dqp:
            for qp in QP_List[:args.num_QP]:
                QP_map = QP_map_base + qp
                qp_map_file = os.path.join(modified_dir + '_QPmap/' + str(qp), os.path.splitext(os.path.basename(img_fname))[0] + '_00000_qp_map.txt')
                np.savetxt(qp_map_file, QP_map, fmt='%d', delimiter=' ')         

hook_0.remove()
hook_1.remove()    
    
  
