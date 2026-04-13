#!/bin/bash
#SBATCH --job-name=hydra_4head_resume_h100_
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=timm.4heads.resume.log
#SBATCH --error=timm.4heads.resume.log

module load python
source $WORK/workspace/venv/bin/activate
echo $SLURM_JOBID  &> timm.4heads.resume.${SLURM_JOBID}.log 
echo "Distributed training: gpu:h100:2"  >> timm.4heads.resume.${SLURM_JOBID}.log 
echo "4 Heads; hidden_dim = 128; resume from epoch 100"  >> timm.4heads.resume.${SLURM_JOBID}.log 
torchrun --master_port 29409 --nproc_per_node=2 main.py --coco_path $WORK/workspace/data/coco --hidden_dim 128 --dim_feedforward 1024 --nheads 4 --epochs 300 --batch_size 8 --device cuda --resume /work/ti5/cdk9132/workspace/output/detr/dist/4heads/5031636/checkpoint0209.pth --start_epoch 210 --output_dir $WORK/workspace/output/detr/dist/4heads/${SLURM_JOBID} >> timm.4heads.resume.${SLURM_JOBID}_2.log 2>&1
