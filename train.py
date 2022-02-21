# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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

#def make_agent(obs_spec, action_spec, cfg):
#    cfg.obs_shape = obs_spec.shape
#    cfg.action_shape = action_spec.shape
#    return hydra.utils.instantiate(cfg)

def make_agent(img_shape, state_shape, action_shape, max_actions, joint_indexes, robot_name, controller_iterations, cfg, device):
    cfg.img_shape = img_shape
    cfg.state_shape = state_shape
    print('MAKING IMAGE with IMG %s STATE %s ACTION %s'%(img_shape, state_shape, action_shape))
    cfg.max_actions = [float(m) for m in max_actions]
    cfg.action_shape = action_shape
    cfg.joint_indexes = [int(ii) for ii in joint_indexes]
    cfg.robot_name = robot_name
    cfg.controller_iterations = controller_iterations
    return hydra.utils.instantiate(cfg)

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.img_shape,
                                self.train_env.state_shape,
                                self.train_env.action_shape,
                                self.train_env.max_actions,
                                self.train_env.joint_indexes,
                                self.train_env.robot_name,
                                self.train_env.controller_iterations,
                                self.cfg.agent,
                                self.cfg.device)
        #self.agent = make_agent(self.train_env.observation_spec(),
        #                        self.train_env.action_spec(),
        #                        self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        if 'robosuite' not in self.cfg.task_name:
            self.task_name = self.cfg.task_name
            self.train_env = dmc.make_dm(self.cfg.task_name, self.cfg.frame_stack,
                                      self.cfg.action_repeat, self.cfg.seed)
            self.eval_env = dmc.make_dm(self.cfg.task_name, self.cfg.frame_stack,
                                     self.cfg.action_repeat, self.cfg.seed)

            self.fps = 20
        else:
            def make_robosuite_env(task_name, use_proprio_obs, env_kwargs, discount, frame_stack=3, seed=1111, randomize=False):
                env = robosuite.make(
                            env_name=task_name,
                            **env_kwargs,
                        )
                if randomize:
                    print('randomizing environment', seed)
                    randomize_color = True
                    randomize_dynamics = False
                    randomize_camera = False
                    randomize_lighting = True
                else:
                    print('normal environment', seed)
                    randomize_color = False
                    randomize_dynamics = False
                    randomize_camera = False
                    randomize_lighting = False

                env = DRQWrapper(env, use_proprio_obs=use_proprio_obs, frame_stack=frame_stack, discount=discount,
                                 randomize_color=randomize_color,
                                 randomize_camera=randomize_camera,
                                 randomize_lighting=randomize_lighting,
                                 randomize_dynamics=randomize_dynamics,
                                 randomize_on_reset=True,
                                 )
                return env

            self.env_kwargs = {}
            self.task_name = self.cfg.env_name
            controller_file = self.cfg.env_override.controller_config_file
            controller_fpath = os.path.join(
                           os.path.split(robosuite.__file__)[0], 'controllers', 'config',
                           controller_file)
            assert os.path.exists(controller_fpath)
            from robosuite.controllers import load_controller_config
            self.env_kwargs['controller_configs'] = load_controller_config(custom_fpath=controller_fpath)
            del self.cfg.env_override.controller_config_file

            for k in self.cfg.env_override:
                self.env_kwargs[k] = self.cfg.env_override[k]
            self.fps = self.env_kwargs['control_freq']
            randomize = False
            if 'randomize' in self.env_kwargs.keys():
                if self.env_kwargs['randomize']:
                    randomize = True

                del self.env_kwargs['randomize']

            self.train_env = make_robosuite_env(
                                            task_name=self.task_name,
                                            use_proprio_obs=self.cfg.use_proprio_obs,
                                            env_kwargs=self.env_kwargs,
                                            discount=self.cfg.discount,
                                            frame_stack=self.cfg.frame_stack,
                                            seed=self.cfg.seed,
                                            randomize=randomize
                                             )
            self.eval_env = make_robosuite_env(
                                           task_name=self.task_name,
                                           use_proprio_obs=self.cfg.use_proprio_obs,
                                           env_kwargs=self.env_kwargs,
                                           discount=self.cfg.discount,
                                           frame_stack=self.cfg.frame_stack,
                                           seed=self.cfg.seed+1,
                                           randomize=randomize
                                            )
            self.train_env.robot_name = self.env_kwargs['robots']

        self.data_specs = (
                      specs.Array(shape=self.train_env.img_shape, dtype=np.uint8, name='img_obs'),
                      specs.Array(shape=self.train_env.state_shape, dtype=np.float32, name='state_obs'),
                      specs.Array(shape=self.train_env.body_shape, dtype=np.float32, name='body'),
                      specs.Array(shape=self.train_env.action_shape, dtype=np.float32, name='action'),
                      specs.Array(shape=(1,), dtype=np.float32, name='reward'),
                      specs.Array(shape=(1,), dtype=np.float32, name='discount'))


        print('setup envs')
        # create replay buffer

        self.replay_storage = ReplayBufferStorage(self.data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            use_specs_names=self.replay_storage._use_specs_names,
            replay_dir=self.work_dir / 'buffer', max_size=self.cfg.replay_buffer_size,
            batch_size=self.cfg.batch_size, num_workers=self.cfg.replay_buffer_num_workers,
            save_snapshot=self.cfg.save_snapshot, nstep=self.cfg.nstep, discount=self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None, fps=self.fps)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def run_eval_agent(self):
        self.replay_storage = ReplayBufferStorage(self.data_specs,
                                                  self.work_dir / 'eval_buffer')

        self.replay_loader_eval = make_replay_loader(
            use_specs_names=self.replay_storage._use_specs_names,
            replay_dir=self.work_dir / 'eval_buffer', max_size=100000,
            batch_size=self.cfg.batch_size, num_workers=1,
            save_snapshot=self.cfg.save_snapshot, nstep=self.cfg.nstep, discount=self.cfg.discount)
        self.eval()


    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.img_obs,
                                            time_step.state_obs,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)

                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame:0>8}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.img_obs)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame:0>8}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.img_obs)
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.img_obs,
                                        time_step.state_obs,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.img_obs)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    else:
        store_python = os.path.join(root_dir, 'python')
        if not os.path.exists(store_python):
            os.makedirs(store_python)
        cmd = 'cp %s %s'%(os.path.join(os.path.split(cur_path)[0], '*.py'), store_python)
        os.system(cmd)

    workspace.train()


if __name__ == '__main__':
    main()
