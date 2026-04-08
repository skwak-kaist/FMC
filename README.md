# FMC: Feature-Guided Machine-Centric Compression

Feature-guided block-wise image blending method for machine-centric image compression, suitable for object detection and instance segmentation tasks.

## Overview

This project implements the method proposed in "Feature-Guided Machine-Centric Image Coding for Downstream Tasks" (IEEE ICMEW 2023). The main idea is to blend original and degraded images using a gradient map of feature loss, enabling compression-friendly images while maintaining machine vision task performance.

**Key Features:**
- Feature-guided preprocessing using task-specific pretrained networks
- Block-wise quality map generation for selective degradation
- No additional training required
- Compatible with VVC (VTM 12.0) and HEVC codecs
- Support for object detection and instance segmentation tasks

**Performance:**
- Average 11% BD-rate gain for object detection
- Average 8% BD-rate gain for instance segmentation
- Compared to MPEG-VCM reference software v0.4

## Project Structure

```
FMIC/
├── 1_create_venv_for_vcm_anchor_detectron2_CUDA11_1_docker.sh  # Environment setup
├── dataset/                          # Dataset directory
│   ├── OpenImages/
│   │   ├── annotations_5k/          # Annotations and image lists
│   │   └── validation/              # Validation images (symlink)
│   ├── TVD/                         # TVD dataset (symlink)
│   ├── Cityscapes/                  # Cityscapes dataset (symlink)
│   └── SFU/                         # SFU dataset (symlink)
├── src/Anchor/                      # Core implementation
│   ├── dataset_modification.py      # Feature-guided preprocessing
│   ├── dataset_conversion.py        # Video encoding/decoding
│   ├── detectron2_predict.py        # Inference using Detectron2
│   └── calculate_bpp.py             # BPP calculation
├── bin/                             # VTM encoder/decoder binaries
├── tools/                           # External tools
│   ├── cocoapi/
│   ├── models/
│   └── ffmpeg-4.2.2-amd64-static/
├── configs/                         # VTM configuration files
├── openimages_improvement_32.sh     # OpenImages experiment script
├── tvd_improvement_64.sh            # TVD experiment script
├── cityscapes_improvement_32.sh     # Cityscapes experiment script
└── sfu_anchor_32.sh                 # SFU anchor script
```

## Dataset Setup

### 1. Dataset Organization

The project supports four datasets: OpenImages, TVD, Cityscapes, and SFU. Use symbolic links to connect actual dataset locations:

```bash
cd dataset/OpenImages
ln -s /path/to/OpenImages/validation validation

cd ../TVD
ln -s /path/to/TVD/TVD_Image_VCM TVD

cd ../Cityscapes
ln -s /path/to/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val validation

cd ../SFU
ln -s /path/to/SFU/SFU-HW-png-VCM SFU-HW-png-VCM
```

### 2. Required Dataset Structure

- **OpenImages**: `dataset/OpenImages/validation/` - Contains validation images
  - Annotations in `dataset/OpenImages/annotations_5k/`
- **TVD**: `dataset/TVD/TVD_Image_VCM/` - TVD image dataset
- **Cityscapes**: `dataset/Cityscapes/validation/` - Cityscapes validation set
- **SFU**: `dataset/SFU/SFU-HW-png-VCM/` - SFU hardware dataset

## Environment Setup

### Prerequisites
- Ubuntu 18.04 or later
- CUDA 11.1, 11.3, or compatible version
- Python 3.8
- VTM (VVC Test Model) encoder/decoder binaries in `bin/` directory

### Installation

1. **Create Python Virtual Environment:**
   ```bash
   ./1_create_venv_for_vcm_anchor_detectron2_CUDA11_1_docker.sh
   ```

   This script will:
   - Create a Python 3.8 virtual environment in `.venv/`
   - Install PyTorch, Detectron2, and dependencies
   - Detect CUDA version and install appropriate Detectron2 version:
     - CUDA 11.1 → Detectron2 v0.4
     - CUDA 11.3 → Detectron2 v0.6
   - Install TensorFlow Object Detection API
   - Download pretrained models (Mask R-CNN, Faster R-CNN)

2. **Install Additional Requirements:**
   ```bash
   source .venv/bin/activate
   pip install -r Requirements.txt
   ```

3. **Prepare VTM Binaries:**
   - Place VTM encoder (`EncoderAppStatic`) and decoder (`DecoderAppStatic`) in `bin/` directory
   - Ensure they are executable: `chmod +x bin/*`

## Running Experiments

Each script follows the same pipeline:
1. **Modification**: Feature-guided preprocessing (blending)
2. **Conversion**: Video encoding (YUV conversion → VTM encoding → decoding)
3. **Prediction**: Run inference using Detectron2
4. **Evaluation**: Calculate mAP metrics
5. **Reporting**: Generate BD-rate reports

### OpenImages Dataset

```bash
./openimages_improvement_32.sh
```

**Parameters:**
- `BLK_SIZE=32`: Block size for quality map (32×32)
- `DS_RATIO=0.25`: Degradation ratio (bicubic downsampling)
- `ALPHA=0.5`: Threshold for quality map blending
- `QP_LIST=(22 27 32 37 42 47)`: Quantization parameters
- `TASK_LIST=("detection" "segmentation")`: Tasks to evaluate

### TVD Dataset

```bash
./tvd_improvement_64.sh
```

**Parameters:**
- `BLK_SIZE=64`: Block size (64×64)
- Gaussian blur for degradation
- Tasks: detection and segmentation

### Cityscapes Dataset

```bash
./cityscapes_improvement_32.sh
```

**Parameters:**
- `BLK_SIZE=32`: Block size (32×32)
- Instance segmentation task

### SFU Dataset (Anchor)

```bash
./sfu_anchor_32.sh
```

**Parameters:**
- Standard VCM anchor without modification
- Detection task only

### Run Mode Configuration

Edit the script to control which stages to run:

```bash
RUN_MODIFICATION=1   # 1: Run feature-guided preprocessing, 0: Skip
RUN_CONVERSION=1     # 1: Run encoding/decoding, 0: Skip
RUN_PREDICTION=1     # 1: Run inference, 0: Skip
RUN_EVALUATION=1     # 1: Run evaluation, 0: Skip
RUN_REPORTING=1      # 1: Generate BD-rate report, 0: Skip
```

## Core Components

### 1. Dataset Modification (`dataset_modification.py`)

Implements feature-guided preprocessing:
- **Input**: Original images + degraded images
- **Feature Extraction**: Uses pretrained Detectron2 backbone (FPN P-layer)
- **Loss Calculation**: L2-norm between original and degraded features
- **Gradient Map**: Back-propagation to obtain importance map
- **Quality Map**: Block-wise merging and normalization
- **Blending**: Selective blending based on quality map threshold

**Usage:**
```bash
python dataset_modification.py \
  --input_dir <input_image_dir> \
  --output_dir <output_blended_dir> \
  --task detection \
  --block_size 32 \
  --alpha 0.5 \
  --degradation_ratio 0.25
```

### 2. Dataset Conversion (`dataset_conversion.py`)

Video encoding and decoding pipeline:
- RGB → YUV 4:2:0 conversion using ffmpeg
- VTM encoding with specified QP
- VTM decoding
- YUV → RGB conversion

**Usage:**
```bash
python dataset_conversion.py \
  --input_dir <input_dir> \
  --output_dir <output_dir> \
  --vtm_encoder <path_to_encoder> \
  --vtm_decoder <path_to_decoder> \
  --qp 27
```

### 3. Inference (`detectron2_predict.py`)

Run object detection or instance segmentation:
- Uses Detectron2 pretrained models
- Supports Faster R-CNN (detection) and Mask R-CNN (segmentation)
- Saves predictions in COCO format

**Usage:**
```bash
python detectron2_predict.py \
  --task detection \
  --input_dir <image_dir> \
  --output_json <output.json> \
  --model faster_rcnn_X_101_32x8d_FPN_3x
```

### 4. BPP Calculation (`calculate_bpp.py`)

Calculate bits-per-pixel from compressed bitstreams.

## Output Structure

```
output/
└── <experiment_name>/
    ├── modified/                # Feature-guided preprocessed images
    ├── converted/               # Encoded/decoded images
    │   ├── QP22/
    │   ├── QP27/
    │   └── ...
    ├── predictions/             # Inference results (JSON)
    ├── evaluations/             # mAP scores
    └── reports/                 # BD-rate reports
```

## Method Pipeline

1. **Input Image** → Degradation function D(·)
2. **Original & Degraded** → Feature extraction F(·) using pretrained encoder
3. **Features** → Loss function L(·) (L2-norm)
4. **Loss** → Back-propagation B(·) → Gradient map
5. **Gradient Map** → Quality map decision Q(·) → Block-wise quality map
6. **Quality Map** → Block-wise blending → Blended image
7. **Blended Image** → VVC Encoder → Bitstream
8. **Bitstream** → VVC Decoder → Decoded image
9. **Decoded Image** → Task Network → Predictions
10. **Predictions** → Evaluation → mAP

## Citation

```bibtex
@inproceedings{kwak2023feature,
  title={Feature-Guided Machine-Centric Image Coding for Downstream Tasks},
  author={Kwak, Sangwoon and Yun, Joungil and Choo, Hyon-Gon and Kim, Munchurl},
  booktitle={2023 IEEE International Conference on Multimedia and Expo Workshops (ICMEW)},
  pages={176--181},
  year={2023},
  organization={IEEE}
}
```

## Authors

- Sangwoon Kwak (s.kwak@etri.re.kr, sw.kwak@kaist.ac.kr)
- Joungil Yun (sigipus@etri.re.kr)
- Hyon-Gon Choo (hyongonchoo@etri.re.kr)
- Munchurl Kim (mkimee@kaist.ac.kr)

## License

This project was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2020-0-00011, Video Coding for Machine).

## References

- MPEG-VCM: [http://mpegx.intevry.fr/software/MPEG/Video/VCM/VCM-RS](http://mpegx.intevry.fr/software/MPEG/Video/VCM/VCM-RS)
- Detectron2: [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
- VTM: [https://github.com/ChristianFeldmann/VTM](https://github.com/ChristianFeldmann/VTM)