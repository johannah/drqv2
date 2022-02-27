#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=63G
#SBATCH --time=23:59:00
#SBATCH --output=/scratch/jhansen/DH22/control/job_%A-%a.out
#SBATCH --array=0-5

unset display

module load python/3.8

source $HOME/.bashrc
source $HOME/johannah/envs/drqv2/bin/activate
export MUJOCO_GL="egl"

experiment=control
seeds=(1000 1001 1002 1003 1004)
num_seeds=5
# environment
tasks=(robosuite_lift_joint_position_img)
cd $HOME/johannah/DH22/drqv2/
python train.py task=robosuite_lift_joint_position_img agent.experiment_type=kine_DH_abs_posquat_controller seed=${seeds[($SLURM_ARRAY_TASK_ID)]}
