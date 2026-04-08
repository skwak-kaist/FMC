#!/bin/bash

# Script for VCM Anchor
# by Sangwoon Kwak (s.kwak@etri.re.kr) and Joungil Yun (sigipus@etri.re.kr)

#set -x
set -e
source .venv/bin/activate


# Number of GPUs available
GPU_NUM=2

# VTM Version (8.2 or 12.0)
# for Nokia Anchor
#VTM_VER=8.2
# for Erricson Anchor
VTM_VER=12.0

# Select Run Mode (1: On, 0: Off)
RUN_CONVERSION=0
RUN_PREDICTION=0
RUN_EVALUATION=0
RUN_REPORTING=1

# Directory Settings
SRC_DIR=$(realpath ./src/OpenImages)
BIN_DIR=$(realpath ./bin)
DATASET_DIR=$(realpath ./dataset/OpenImages)
ANNO_DIR=$DATASET_DIR/annotations_5k
OUTPUT_DIR=./output/OpenImages_test_dqp1
mkdir -p ${OUTPUT_DIR}
OUTPUT_DIR=$(realpath $OUTPUT_DIR)

## original source image path (jpg)
SOURCE_DATA_DIR=$DATASET_DIR/validation


# get TF object_detection path
TF_OD_DIR=$(python -c "import object_detection,os;print(os.path.dirname(object_detection.__file__))")

# Task, QP, Down-scaling Params
TASK_LIST=("detection")
DO_UNCMPRD=0
QP_LIST=(22 27 32 37)
#QP_LIST=(47)

# for Nokia Anchor
#DS_LIST=(2)
# for Erricson Anchor
DS_LIST=(0 1 2)

DS2SCALE=("100" "075" "050" "025")



COCO_CLASSES_FILE=$ANNO_DIR/coco_classes.txt
SELECTED_CLASSES=$ANNO_DIR/selected_classes.txt



# 전체 Task에 대한 결과를 모아서 출력하도록 분리함
# Reporting
if [ $RUN_REPORTING == 1 ]
then
    log_fname=$OUTPUT_DIR/log.txt
    echo "Results" > $log_fname

    for task in "${TASK_LIST[@]}"
    do
        encoded_bitstream_dir=$OUTPUT_DIR/${task}/modified_bitstream

        jpg_input_list=$OUTPUT_DIR/${task}_validation_input_jpg.lst

        input_fname=$ANNO_DIR/${task}_validation_input_5k.lst
        cat $input_fname | sed 's/png/jpg/' > $jpg_input_list

        echo >> $log_fname
        echo "Evaluation result for ${task} : " >> $log_fname

        if [ $DO_UNCMPRD == 1 ]
        then
            output_metrics=$OUTPUT_DIR/${task}_uncmprd_metric.txt
            rslt=$(head $output_metrics -n 1 | cut -d , -f 2)
            echo "uncmprd ${rslt}" >> $log_fname
        fi

        for ds in "${DS_LIST[@]}"
        do
            for qp in "${QP_LIST[@]}"
            do
                scale=${DS2SCALE[ds]}
                output_metrics=$OUTPUT_DIR/${task}_QP${qp}_${scale}_metric.txt
                rslt=$(head $output_metrics -n 1 | cut -d , -f 2)
                echo $qp $scale $rslt >> $log_fname
            done
        done

        echo >> $log_fname
        echo "BPP result for ${task} : " >> $log_fname

        for ds in "${DS_LIST[@]}"
        do
            for qp in "${QP_LIST[@]}"
            do
                scale=${DS2SCALE[ds]}
                python -u $SRC_DIR/calculate_bpp.py \
                    --input_dir ${SOURCE_DATA_DIR} \
                    --bitstream_dir ${encoded_bitstream_dir}/QP${qp}_${scale} \
                    --input_fname ${jpg_input_list} \
                    --output_data_file $(realpath $OUTPUT_DIR/${task}_QP${qp}_${scale}_bpp.csv) \
                    --qp ${qp} \
                    --ds_level ${ds} >> $log_fname
            done
        done
    done
fi
echo 
echo "All Done!"

echo

exit 0

