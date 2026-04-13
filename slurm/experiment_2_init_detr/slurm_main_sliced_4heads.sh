#!/bin/bash
#SBATCH --job-name=hydra_sliced_4head_2_h100_
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=timm.4heads.sliced.log
#SBATCH --error=timm.4heads.sliced.log

module load python
source $WORK/workspace/venv/bin/activate
echo $SLURM_JOBID  &> timm.4heads.sliced.${SLURM_JOBID}.log 
echo "Distributed training: gpu:h100:2"  >> timm.4heads.sliced.${SLURM_JOBID}.log 
echo "Experiment 1: Naive Sliced DETR"  >> timm.4heads.sliced.${SLURM_JOBID}.log 
echo "4 Heads; hidden_dim = 128 "  >> timm.4heads.sliced.${SLURM_JOBID}.log 
echo "Scalable DETR"  >> timm.4heads.sliced.${SLURM_JOBID}.log 
torchrun --master_port 29406 --nproc_per_node=2 sliced_main.py --coco_path $WORK/workspace/data/coco --hidden_dim 128 --world_size 2 --dim_feedforward 1024 --nheads 4 --epochs 300 --batch_size 8 --device cuda --start_epoch 190 --output_dir $WORK/workspace/output/detr/dist/4heads/sliced/${SLURM_JOBID} >> timm.4heads.sliced.${SLURM_JOBID}.log 2>&1