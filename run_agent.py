import os
import torch

torch.set_num_threads(2)
cur_path = os.path.abspath(__file__)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from dm_env import specs
from pathlib import Path
from robosuite.controllers import load_controller_config
from robosuite.utils import transform_utils
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

def run_agent(w, n_episodes=1):
    step, episode, total_reward = 0, 0, 0
    video_recorder = TrainVideoRecorder(w.work_dir, fps=w.fps, save_dir_name='run_eval_video')
    for e in range(n_episodes):
        time_step = w.eval_env.reset()
        video_recorder.init(time_step.img_obs, enabled=True)
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(w.agent):
                action = w.agent.act(time_step.img_obs,
                                        time_step.state_obs,
                                        w.global_step,
                                        eval_mode=True)
            time_step = w.eval_env.step(action)

            video_recorder.record(time_step.img_obs)
            total_reward += time_step.reward
            step += 1

        episode += 1
        video_recorder.save(f'{w.global_frame:0>8}_{e:0>3}.mp4')

    print(total_reward/episode)

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
    run_agent(workspace, 3)

if __name__ == '__main__':
    run_main()
