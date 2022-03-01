#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=63G
#SBATCH --time=47:59:00
#SBATCH --output=/scratch/jhansen/DH22/lift/job_%A-%a.out
#SBATCH --array=0-25

unset display

module load python/3.8

source $HOME/.bashrc
source $HOME/johannah/envs/drqv2/bin/activate
export MUJOCO_GL="egl"

experiment=lift
seeds=(1000 1001 1002 1003 1004)
num_seeds=5
# environment
exps=(none body kine_DH_abs_posquat kine_DH_abs_posquat_detach kine_DH_abs_posquat_control)
cd $HOME/johannah/DH22/drqv2/
python train.py task=robosuite_lift_joint_position_img agent.experiment_type=${exps[(($SLURM_ARRAY_TASK_ID / $num_seeds))]} seed=${seeds[(($SLURM_ARRAY_TASK_ID % $num_seeds))]}
