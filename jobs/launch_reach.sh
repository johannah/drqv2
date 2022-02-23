#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=124G
#SBATCH --time=71:59:00
#SBATCH --output=/scratch/jhansen/DH22/reach/job_%A-%a.out
#SBATCH --array=0-20

unset display

module load opencv
module load python/3.8

source $HOME/.bashrc
source $HOME/johannah/envs/drqv2/bin/activate
export MUJOCO_GL="egl"

experiment=reach
seeds=(1000 1001 1002 1003 1004)
num_seeds=5
# environment
tasks=(robosuite_random_reach_joint_position_img robosuite_random_reach_joint_position_img_body robosuite_random_reach_joint_position_img_DH_abs_posquat_control robosuite_random_reach_joint_position_img_DH_abs_posquat_control_body)
cd $HOME/johannah/DH22/drqv2/
python train.py task=${tasks[(($SLURM_ARRAY_TASK_ID / $num_seeds))]} seed=${seeds[(($SLURM_ARRAY_TASK_ID % $num_seeds))]}