#!/bin/bash

source experiments/path.sh

expName="KDE_SD"
cfgDir="$reposDir/fovea/configs"
nGPU=2

python tools/compute_dataset_saliency.py \
    --data-dir $dataDir \
    --out-dir "$outDir/$expName" \
    --dataset-config "$cfgDir/datasets/avhd.py" \
    --model-config "$cfgDir/models/fovea_faster_rcnn_r50_fpn.py" \
    --schedule-config "$cfgDir/schedules/schedule_poly.py" \
    --runtime-config "$cfgDir/default_runtime.py" \
    --use-fovea \
    --grid-net-cfg "$cfgDir/models/plain_kde_grid.py" \

# Off the shelf model
python tools/test.py \
    --gpu-test-pre \
    --data-dir $dataDir \
    --out-dir "$outDir/$expName/test_OTS" \
    --map-classes \
    --load-task-weights "$checkpointsDir/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth" \
    --dataset-config "$cfgDir/datasets/avhd.py" \
    --model-config "$cfgDir/models/fovea_faster_rcnn_r50_fpn.py" \
    --schedule-config "$cfgDir/schedules/schedule_poly.py" \
    --runtime-config "$cfgDir/default_runtime.py" \
    --preprocess-scale "1" \
    --reg-decoded-bbox \
    --use-fovea \
    --grid-net-cfg "$cfgDir/models/fixed_kde_grid.py" \
    --saliency-file "$outDir/$expName/dataset_saliency.pkl" \

# Finetuned model
python tools/test.py \
    --gpu-test-pre \
    --data-dir $dataDir \
    --out-dir "$outDir/$expName/test_FT" \
    --weights "$checkpointsDir/KDE_SD.pth" \
    --dataset-config "$cfgDir/datasets/avhd.py" \
    --model-config "$cfgDir/models/fovea_faster_rcnn_r50_fpn.py" \
    --schedule-config "$cfgDir/schedules/schedule_poly.py" \
    --runtime-config "$cfgDir/default_runtime.py" \
    --preprocess-scale "1" \
    --reg-decoded-bbox \
    --num-classes 8 \
    --use-fovea \
    --grid-net-cfg "$cfgDir/models/fixed_kde_grid.py" \
    --saliency-file "$outDir/$expName/dataset_saliency.pkl" \
    --vis-options \
        input_image="$outDir/$expName/test_FT/vis/input_image" \
        warped_image="$outDir/$expName/test_FT/vis/warped_image" \
        saliency="$outDir/$expName/test_FT/vis/saliency" \
