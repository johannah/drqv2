import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import torch

torch.set_num_threads(2)
cur_path = os.path.abspath(__file__)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from copy import deepcopy
from dm_env import specs
from pathlib import Path
from robosuite.controllers import load_controller_config
from robosuite.utils import transform_utils
from robosuite.wrappers.sim2real_wrapper import JacoSim2RealWrapper
from robosuite_wrapper import DRQWrapper
from typing import NamedTuple
import robosuite


import hydra
import numpy as np
np.set_printoptions(suppress=True)

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from IPython import embed

torch.backends.cudnn.benchmark = True


def run_clone_agent(w, n_episodes=1):
    frames = []
    robot_on = False
    joint_positions = []
    robot_joint_positions = []
    predicted_joint_positions = []
    step, episode, total_reward = 0, 0, 0
    video_recorder = TrainVideoRecorder(w.work_dir, fps=w.fps, save_dir_name='clone_run_eval_video')
    robot_video_recorder = TrainVideoRecorder(w.work_dir, fps=w.fps, save_dir_name='clone_run_eval_video')
    if robot_on:
        robot_env = JacoSim2RealWrapper(w.train_env.env, robot_server_port=9031)
    time_step = w.eval_env.reset()
    video_recorder.init(time_step.img_obs, enabled=True)
    for e in range(n_episodes):
        time_step = w.eval_env.reset()
        episode_step = 0; episode_reward = 0
        if robot_on:
            robot_time_step = robot_env.reset()
        robot_video_recorder.init(time_step.img_obs, enabled=robot_on)
        keep_going = True
        joint_pos = deepcopy(w.eval_env.sim.data.qpos[w.eval_env.robots[0]._ref_joint_pos_indexes])
        last_joint_pos = deepcopy(w.eval_env.sim.data.qpos[w.eval_env.robots[0]._ref_joint_pos_indexes])
        while not time_step.last() and keep_going:
            joint_positions.append(joint_pos)
            with torch.no_grad(), utils.eval_mode(w.agent):
                action = w.agent.act(time_step.img_obs,
                                        time_step.state_obs,
                                        w.global_step,
                                        eval_mode=True)
            time_step = w.eval_env.step(action)
            #print('agent', episode_step, action)
            predicted_joint_positions.append(deepcopy(joint_pos+action[:len(joint_pos)]))
            last_joint_pos = deepcopy(joint_pos)
            joint_pos = deepcopy(w.eval_env.sim.data.qpos[w.eval_env.robots[0]._ref_joint_pos_indexes])
            real_joint_action = list(joint_pos - last_joint_pos)
            #print('real', real_joint_action)
            if robot_on:
                robot_time_step = robot_env.step(real_joint_action + [0])
                robot_joint_pos = deepcopy(time_step[0]['robot0_joint_pos'])
                robot_joint_positions.append(robot_joint_pos)
                #robot_img = deepcopy(time_step[0]['nearfrontview_image'])
                robot_video_recorder.record(time_step.img_obs)
            video_recorder.record(time_step.img_obs)
            episode_reward += time_step.reward
            if episode_step > 5:
                keep_going = False
            if w.eval_env._check_success():
                keep_going = False
            episode_step += 1

        total_reward += episode_reward
        step += episode_step
        print('episode', e, episode_step, step, episode_reward)
    video_recorder.save(f'{w.global_frame:0>8}_{e:0>3}_all.mp4')
    if robot_on:
        robot_video_recorder.save(f'{w.global_frame:0>8}_{e:0>3}_robot_all.mp4')
    joint_positions = np.array(joint_positions)
    predicted_joint_positions = np.array(predicted_joint_positions)
    f,ax = plt.subplots(joint_positions.shape[1], figsize=(15,20))
    for i in range(joint_positions.shape[1]):
        ax[i].plot(joint_positions[:,i])
        ax[i].plot(predicted_joint_positions[:,i])
        ax[i].set_title('joint %d'%i)
        ax[i].set_ylim(-2*np.pi, 2*np.pi)
    plt.savefig('joints.png')
    print('mean reward', total_reward/n_episodes)

def run_agent(w, n_episodes=1):
    step, episode, total_reward = 0, 0, 0
    video_recorder = TrainVideoRecorder(w.work_dir, fps=w.fps, save_dir_name='run_eval_video')
    video_recorder.init(time_step.img_obs, enabled=True)
    for e in range(n_episodes):
        episode_reward = 0
        time_step = w.eval_env.reset()
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(w.agent):
                action = w.agent.act(time_step.img_obs,
                                        time_step.state_obs,
                                        w.global_step,
                                        eval_mode=True)
            time_step = w.eval_env.step(action)

            video_recorder.record(time_step.img_obs)
            episode_reward += time_step.reward
            step += 1

        total_reward += episode_reward
        print('episode', e, episode_reward)
        episode += 1
    video_recorder.save(f'{w.global_frame:0>8}_{e:0>3}_all.mp4')

    print('mean reward', total_reward/n_episodes)

@hydra.main(config_path='cfgs', config_name='config')
def run_main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    print('root dir is', root_dir)
    print(workspace.cfg)
    assert snapshot.exists()
    print(f'resuming: {snapshot}')
    workspace.load_snapshot()
    #run_agent(workspace, 3)
    run_clone_agent(workspace, 8)

if __name__ == '__main__':
    run_main()
