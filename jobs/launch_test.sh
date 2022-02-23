#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=0:00:02
#SBATCH --output=testjob_%A-%a.out
#SBATCH --array=0-20

unset display

module load opencv
module load python/3.8

source $HOME/.bashrc
source $HOME/johannah/envs/drqv2/bin/activate
export MUJOCO_GL="egl"
seeds=(1000 1001 1002 1003 1004)
num_seeds=5
experiment=ours

# environment
tasks=(robosuite_random_reach_joint_position_img robosuite_random_reach_joint_position_img_body robosuite_random_reach_joint_position_img_DH_abs_posquat_control robosuite_random_reach_joint_position_img_DH_abs_posquat_control_body)
#cd $HOME/johannah/DH22/drqv2/
cd $HOME/johannah/
echo 'hello' task=${tasks[(($SLURM_ARRAY_TASK_ID / $num_seeds))]} seed=${seeds[(($SLURM_ARRAY_TASK_ID % $num_seeds))]}

