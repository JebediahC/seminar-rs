#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -o log_%j.log                   # File to store standard output
#SBATCH -e log_%j.log                   # File to store standard error
#SBATCH --time=3:00:00                 # Set a time limit
#SBATCH --container-image=docker://nvcr.io/nvidia/pytorch:25.11-py3
#SBATCH --container-mounts=/dss/dsshome1/02/di97tod/erc-work-data:/dss/dsshome1/02/di97tod/erc-work-data

echo "Start on $(hostname) at $(date)"  # Run outside of srun

unset SLURM_JOB_IDi

export SPLIT=small_1_32

# enroot import docker://nvcr.io/nvidia/pytorch:25.11-py3

# enroot create ~/nvidia+pytorch+25.11-py3.sqsh
# enroot start --root --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all nvidia+pytorch+25.11-py3

cd $HOME/seminar-rs/UniMatch-V2/

echo "NVIDIA_DRIVER_CAPABILITIES=compute,utility" >> /etc/environment
echo "NVIDIA_VISIBLE_DEVICES=all" >> /etc/environment

nvidia-smi

sh scripts/train.sh

echo "End on $(hostname) at $(date)"    # Run outside of srun