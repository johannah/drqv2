#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=63G
#SBATCH --time=71:59:00
#SBATCH --output=/scratch/jhansen/DH22/reach_random/job_%A-%a.out
#SBATCH --array=1000-1004

unset display

module load python/3.8

source $HOME/.bashrc
source $HOME/johannah/envs/drqv2/bin/activate
export MUJOCO_GL="egl"

experiment=random_reach_10hz
# environment
cd $HOME/johannah/DH22/drqv2/
python train.py task=robosuite_reach_joint_position_10hz_img_randomize agent.experiment_type=kine_DH_abs_posquat seed=$SLURM_ARRAY_TASK_ID 
