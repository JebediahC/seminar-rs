#!/bin/bash

# Clone repository (if not already cloned)
git clone https://github.com/JebediahC/seminar-rs.git
cd seminar-rs

# Download and setup submodules
git submodule update --init --recursive
cd UniMatch-V2/

# Download pretrained models
mkdir -p pretrained_models
wget -P pretrained_models https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth # DINOv2-Base

# Link pretrained model to expected location
mkdir -p pretrained
ln -sf ../pretrained_models/dinov2_vits14_pretrain.pth pretrained/dinov2_base.pth
# Setup gf7-building dataset (adjust paths to your local COCO installation)
mkdir -p gf7-building
ln -sf /path/to/your/gf-7-building-4bands/Train gf7-building/Train
ln -sf /path/to/your/gf-7-building-4bands/Val gf7-building/Val
ln -sf /path/to/your/gf-7-building-4bands/Test gf7-building/Test

# Custom configuration for training
dataset='gf7-building'
method='unimatch_v2'
exp='dinov2_base'
split='small_1_32'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

# Train with specified number of GPUs (default: 1) and port (default: 29500)
NUM_GPUS=${1:-1}
PORT=${2:-29500}

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=localhost \
    --master_port=$PORT \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $PORT 2>&1 | tee $save_path/out.log