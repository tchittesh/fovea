#!/bin/bash

source experiments/path.sh

expName="LKDE_SI"
cfgDir="$reposDir/fovea/configs"
nGPU=2

# Train model
# This validates using ground truth annotations instead of previous frame detections, which will overestimate performance. 
python -m torch.distributed.launch \
	--nproc_per_node $nGPU \
	tools/train.py \
    --data-dir "$dataDir" \
    --work-dir "$outDir/$expName/work" \
    --load-task-weights "$checkpointsDir/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth" \
    --gpus $nGPU \
    --launcher pytorch \
    --dataset-config "$cfgDir/datasets/avhd_gt.py" \
    --model-config "$cfgDir/models/fovea_faster_rcnn_r50_fpn.py" \
    --schedule-config "$cfgDir/schedules/schedule_poly.py" \
    --runtime-config "$cfgDir/default_runtime.py" \
    --preprocess-scale "1" \
    --num-classes 8 \
    --reg-decoded-bbox \
    --use-fovea \
    --grid-net-cfg "$cfgDir/models/learned_kde_grid.py" \
    --gridnet-lr-mult 0.005 \
    --gridnet-wd-mult 0.0 \

# Test trained model
python tools/test.py \
    --gpu-test-pre \
    --use-prevdets \
    --data-dir $dataDir \
    --out-dir "$outDir/$expName/test_trained" \
    --weights "$outDir/$expName/latest.pth" \
    --dataset-config "$cfgDir/datasets/avhd_prevdet.py" \
    --model-config "$cfgDir/models/fovea_faster_rcnn_r50_fpn.py" \
    --schedule-config "$cfgDir/schedules/schedule_poly.py" \
    --runtime-config "$cfgDir/default_runtime.py" \
    --preprocess-scale "1" \
    --reg-decoded-bbox \
    --num-classes 8 \
    --use-fovea \
    --grid-net-cfg "$cfgDir/models/learned_kde_grid.py" \
    --gridnet-lr-mult 0.005 \
    --gridnet-wd-mult 0.0 \

# Finetuned model
python tools/test.py \
    --gpu-test-pre \
    --use-prevdets \
    --data-dir $dataDir \
    --out-dir "$outDir/$expName/test_FT" \
    --weights "$checkpointsDir/LKDE_SI.pth" \
    --dataset-config "$cfgDir/datasets/avhd_prevdet.py" \
    --model-config "$cfgDir/models/fovea_faster_rcnn_r50_fpn.py" \
    --schedule-config "$cfgDir/schedules/schedule_poly.py" \
    --runtime-config "$cfgDir/default_runtime.py" \
    --preprocess-scale "1" \
    --reg-decoded-bbox \
    --num-classes 8 \
    --use-fovea \
    --grid-net-cfg "$cfgDir/models/learned_kde_grid.py" \
    --gridnet-lr-mult 0.005 \
    --gridnet-wd-mult 0.0 \
    --vis-options \
        input_image="$outDir/$expName/test_FT/vis/input_image" \
        warped_image="$outDir/$expName/test_FT/vis/warped_image" \
        saliency="$outDir/$expName/test_FT/vis/saliency" \
        magnification_heatmap="$outDir/$expName/test_FT/vis/magnification_heatmap" \
