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
# for dQP VTM
#VTM_VER=dqp
dqp=0

# Select Run Mode (1: On, 0: Off)
RUN_MODIFICATION=1
RUN_CONVERSION=1
RUN_PREDICTION=1
RUN_EVALUATION=1
RUN_REPORTING=1

# Directory Settings
SRC_DIR=$(realpath ./src/Anchor)
BIN_DIR=$(realpath ./bin)
DATASET_DIR=$(realpath ./dataset/TVD)
ANNO_DIR=$DATASET_DIR/annotations
OUTPUT_DIR=./output/TVD_b5
mkdir -p ${OUTPUT_DIR}
OUTPUT_DIR=$(realpath $OUTPUT_DIR)

## original source image path (jpg)
SOURCE_DATA_DIR=$DATASET_DIR/validation


# get TF object_detection path
TF_OD_DIR=$(python -c "import object_detection,os;print(os.path.dirname(object_detection.__file__))")

# Task, QP, Down-scaling Params
#TASK_LIST=("detection" "segmentation")
TASK_LIST=("detection")
DO_UNCMPRD=0
QP_LIST=(27 32 37)

# for Nokia Anchor
#DS_LIST=(0)
# for Erricson Anchor
#DS_LIST=(0 1 2 3)
DS_LIST=(0 1)

DS2SCALE=("100" "075" "050" "025")
BLK_SIZE=32
DS_RATIO=0.25
ALPHA=0.4
filter_type=1
loss_type=2

num_QP=3


COCO_CLASSES_FILE=$ANNO_DIR/coco_classes.txt
SELECTED_CLASSES=$ANNO_DIR/selected_classes_tvd.txt


# Function : Dataset Modification for VCM
function process_modification() {
    local param_task=$1
    local param_task_info=$2
    IFS=',' read pram_input_dir param_modified_dir param_input_fname <<< "${param_task_info}"

    echo
    echo "Processing Dataset Modification for ${param_task} ..."

    param_input_fname=$(realpath $param_input_fname)
    echo ${param_input_fname}

    python -u $SRC_DIR/dataset_modification.py \
        --input_dir ${pram_input_dir} \
        --modified_dir ${param_modified_dir} \
        --task ${param_task} \
        --input_file ${param_input_fname} \
        --block_size ${BLK_SIZE} \
        --degrade_level ${DS_RATIO} \
        --alpha ${ALPHA} \
        --filter_type ${filter_type} \
        --loss_type ${loss_type} \
        --num_QP ${num_QP} \
        --dqp ${dqp}

    echo "Dataset Modification for ${param_task} done."
    echo 
}
export -f process_modification


# Function : Dataset Conversion by VTM 8.2 or 12.0
function process_conversion() {
    local param_task=$1
    local param_task_info=$2
    IFS=',' read param_task_id param_input_dir param_converted_dir param_bitstream_dir param_qpmap_dir param_input_file param_qp param_ds <<< "${param_task_info}"

    echo
    echo "Processing Dataset Conversion for ${param_task} ${param_task_id} ..."

    param_input_dir=$(realpath $param_input_dir)
    #param_converted_dir=$(realpath $param_converted_dir)
    #param_bitstream_dir=$(realpath $param_bitstream_dir)
    param_input_file=$(realpath $param_input_file)
    echo ${param_input_dir}
    echo ${param_converted_dir}
    echo ${param_bitstream_dir}
    echo ${param_input_file}


    if [ $dqp != 0 ]
    then
        ln -s ${param_qpmap_dir}/${param_qp} qp
        
        python -u $SRC_DIR/dataset_conversion_dqp.py \
            --input_dir ${param_input_dir} \
            --converted_dir ${param_converted_dir} \
            --bitstream_dir ${param_bitstream_dir} \
            --qpmap_dir ${param_qpmap_dir} \
            --input_file ${param_input_file} \
            --qp ${param_qp} \
            --ds_level ${param_ds} \
            --vtm_ver ${VTM_VER} \
            --bin_dir ${BIN_DIR} \
            --num_QP ${num_QP} \
            --dqp ${dqp}

        rm -r qp

    else
        python -u $SRC_DIR/dataset_conversion.py \
            --input_dir ${param_input_dir} \
            --converted_dir ${param_converted_dir} \
            --bitstream_dir ${param_bitstream_dir} \
            --input_file ${param_input_file} \
            --qp ${param_qp} \
            --ds_level ${param_ds} \
            --vtm_ver ${VTM_VER} \
            --bin_dir ${BIN_DIR}
    fi

    echo "Dataset Conversion for ${param_task} ${param_task_id} done."
    echo 
}
export -f process_conversion


# Function : Prediction by Detection2
function process_prediction() {
    local param_task=$1
    local param_task_info=$2
    IFS=',' read param_task_id param_input_dir param_input_fname param_output_fname param_cuda_dev <<< "${param_task_info}"

    echo
    echo "Processing Prediction for ${param_task} ${param_task_id} ..."

    param_input_dir=$(realpath $param_input_dir)
    param_input_fname=$(realpath $param_input_fname)
    #param_output_fname=$(realpath $param_output_fname)
    echo ${param_input_dir}
    echo ${param_input_fname}
    echo ${param_output_fname}

    CUDA_VISIBLE_DEVICES=${param_cuda_dev} python -u $SRC_DIR/detectron2_predict.py \
        --input_dir ${param_input_dir} \
        --task ${param_task} \
        --input_file ${param_input_fname} \
        --output_file ${param_output_fname} \
        --coco_classes_file ${COCO_CLASSES_FILE}

    echo "Creating OID expanded results ..."

    python -u $SRC_DIR/cvt_detectron_coco_oid.py \
        --coco_output_file ${param_output_fname} \
        --oid_output_file ${param_output_fname}.oid.txt \
        --selected_classes ${SELECTED_CLASSES}

    echo "Prediction for ${param_task} ${param_task_id} done."
    echo
}
export -f process_prediction


# Function : Evaluation by tensorflow object detection API
function process_evaluation() {
    local param_task=$1
    local param_task_info=$2
    IFS=',' read param_task_id <<< "${param_task_info}"

    echo
    echo "Processing Evaluation for ${param_task} ${param_task_id} ..."
    
    # [주의!] detection인 경우 nokia 제공 annotaion_5k 파일명 "detection_validation_5k_bbox.csv"이 수정 적용됨
    local param_bounding_boxes=$ANNO_DIR/${param_task}_validation_bbox.csv 
    local param_image_labels=$ANNO_DIR/${param_task}_validation_labels.csv
    local param_input_predictions=$OUTPUT_DIR/output_${param_task}_${param_task_id}.txt.oid.txt
    local param_output_metrics=$OUTPUT_DIR/${param_task}_${param_task_id}_metric.txt
    local param_label_map=$ANNO_DIR/coco_label_map.pbtxt
    if [ $task == "detection" ]
    then
        local param_hierarchy_file=$ANNO_DIR/label47_hierarchy.json

        python -u $TF_OD_DIR/metrics/oid_challenge_evaluation.py \
            --input_annotations_boxes=${param_bounding_boxes} \
            --input_annotations_labels=${param_image_labels} \
            --input_class_labelmap=${param_label_map} \
            --input_predictions=${param_input_predictions} \
            --output_metrics=${param_output_metrics}

    else
        local param_instance_segmentations=$ANNO_DIR/segmentation_validation_masks.csv
        local param_segmentation_mask_dir=$ANNO_DIR/tvd_validation_masks
        local param_annotation_resized=$OUTPUT_DIR/segmentation_${param_task_id}_resized.txt

        python -u $SRC_DIR/gen_gt_resized.py \
            --input_annotations=${param_instance_segmentations} \
            --gt_mask_dir=${param_segmentation_mask_dir} \
            --input_predictions=${param_input_predictions} \
            --output_annotations=${param_annotation_resized}

        python -u $TF_OD_DIR/metrics/oid_challenge_evaluation.py \
            --input_annotations_boxes=${param_bounding_boxes} \
            --input_annotations_labels=${param_image_labels} \
            --input_class_labelmap=${param_label_map} \
            --input_predictions=${param_input_predictions} \
            --input_annotations_segm=${param_annotation_resized} \
            --output_metrics=${param_output_metrics}
    fi

    echo "Evaluation for ${param_task} ${param_task_id} done."
    echo 
}
export -f process_evaluation


for task in "${TASK_LIST[@]}"
do
    modified_data_dir=$OUTPUT_DIR/${task}/modified
    converted_data_dir=$OUTPUT_DIR/${task}/modified_converted
    encoded_bitstream_dir=$OUTPUT_DIR/${task}/modified_bitstream

    qpmap_dir=$OUTPUT_DIR/${task}/modified_QPmap

    jpg_input_list=$OUTPUT_DIR/${task}_validation_input_jpg.lst

    input_fname=$ANNO_DIR/${task}_validation_input.lst
    cat $input_fname | sed 's/png/jpg/' > $jpg_input_list

    hierarchy_file=$ANNO_DIR/label50_hierarchy.json
    bounding_boxes=$ANNO_DIR/detection_validation_bbox.csv
    image_labels=$ANNO_DIR/detection_validation_labels.csv
    label_map=$ANNO_DIR/coco_label_map.pbtxt


    ## Modification
    if [ $RUN_MODIFICATION == 1 ]
    then
        task_info_list=("${SOURCE_DATA_DIR}","${modified_data_dir}","${jpg_input_list}")
        for task_info in "${task_info_list[@]}"
        do
            process_modification ${task} ${task_info}
        done
    fi


    ## Conversion
    if [ $RUN_CONVERSION == 1 ]
    then
        task_info_list=()

        for ds in "${DS_LIST[@]}"
        do
            for qp in "${QP_LIST[@]}"
            do
                scale=${DS2SCALE[ds]}
                task_info_list+=("QP${qp}_${scale}","${modified_data_dir}","${converted_data_dir}","${encoded_bitstream_dir}","${qpmap_dir}","${input_fname}",${qp},${ds})

            done
        done
         
        for task_info in "${task_info_list[@]}"
        do
            process_conversion ${task} ${task_info}
        done
    fi


    # Prediction
    if [ $RUN_PREDICTION == 1 ]
    then
        task_info_list=()
        cuda_dev=0


        if [ $DO_UNCMPRD == 1 ]
        then
            output_fname=$OUTPUT_DIR/output_${task}_uncmprd.txt
            task_info_list+=("uncmprd","${modified_data_dir}","${input_fname}","${output_fname}",${cuda_dev})
            cuda_dev=$(($cuda_dev+1))
            cuda_dev=$(($cuda_dev%$GPU_NUM))
        fi

        for ds in "${DS_LIST[@]}"
        do
            for qp in "${QP_LIST[@]}"
            do
                scale=${DS2SCALE[ds]}
                output_fname=$OUTPUT_DIR/output_${task}_QP${qp}_${scale}.txt
                task_info_list+=("QP${qp}_${scale}","${converted_data_dir}/QP${qp}_${scale}","${input_fname}","${output_fname}",${cuda_dev})    
                cuda_dev=$(($cuda_dev+1))
                cuda_dev=$(($cuda_dev%$GPU_NUM))
            done
        done



        for task_info in "${task_info_list[@]}"
        do
            process_prediction ${task} ${task_info}
        done
    fi


    # Evaluation
    if [ $RUN_EVALUATION == 1 ]
    then
        task_info_list=()

        if [ $DO_UNCMPRD == 1 ]
        then
            task_info_list+=("uncmprd","${TF_od_dir}")
        fi

        for ds in "${DS_LIST[@]}"
        do
            for qp in "${QP_LIST[@]}"
            do
                scale=${DS2SCALE[ds]}
                task_info_list+=("QP${qp}_${scale}","${TF_od_dir}")
            done
        done

        for task_info in "${task_info_list[@]}"
        do
            process_evaluation ${task} ${task_info}
        done        
    fi
done


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

        input_fname=$ANNO_DIR/${task}_validation_input.lst
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

