# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import dm_env
from pathlib import Path
from dm_env import specs
import hydra
import numpy as np
np.set_printoptions(suppress=True)

from copy import deepcopy
import torch
#from dmc import ExtendedTimeStepWrapper
#import dmc
import h5py
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robosuite.controllers import load_controller_config
from robosuite.utils import transform_utils
from robosuite.wrappers import DRQDHImageDomainRandomizationWrapper, ExtendedTimeStep
from typing import NamedTuple
import robosuite

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from IPython import embed
from collections import deque
torch.backends.cudnn.benchmark = True


def make_agent(obs_shape, action_shape, max_action, robot_name, cfg):
    cfg.obs_shape = obs_shape
    cfg.max_action = float(max_action)
    cfg.action_shape = action_shape
    cfg.robot_name = robot_name
    return hydra.utils.instantiate(cfg)


def make_robosuite_env(task_name, xpos_targets, env_kwargs, discount, frame_stack=3, seed=1111):
    env = robosuite.make(
                env_name=task_name,
                **env_kwargs,
            )
    env = DRQDHImageDomainRandomizationWrapper(env, xpos_targets=xpos_targets, frame_stack=frame_stack, discount=discount)
    return env


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.obs_shape,
                                self.train_env.action_shape,
                                self.train_env.action_spec[1][0], # HACKY
                                self.train_env.robots[0].name,
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        #dummy_spec = dict(
        #     obs=dict(
        #         low_dim=[
        #                  'robot0_joint_pos', 'robot0_joint_pos_cos',
        #                   'robot0_joint_pos_sin', 'robot0_joint_vel',
        #                   'robot0_eef_pos', 'robot0_eef_quat',
        #                   'robot0_gripper_qpos', 'robot0_gripper_qvel',
        #                   'Can_pos', 'Can_quat',
        #                   'Can_to_robot0_eef_pos', 'Can_to_robot0_eef_quat',
        #                   'robot0_proprio-state', 'object-state'],
        #         rgb=['frontview_image', 'sideview_image', 'agentview_image', 'birdview_image]
        #     ))
        dummy_spec = dict(
             obs=dict(
                 low_dim=[],
                 rgb=['frontview_image']
             ))

        #ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
        if self.cfg.dataset_path == '':
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
        else:
            env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=self.cfg.dataset_path)
            self.env_kwargs = env_meta['env_kwargs']
            self.task_name = env_meta['env_name']
        for k in self.cfg.env_override:
            self.env_kwargs[k] = self.cfg.env_override[k]
        if 'has_renderer' in self.env_kwargs:
            del self.env_kwargs['has_renderer']
        self.xpos_targets = self.cfg.xpos_targets
        self.train_env = make_robosuite_env(
                                            task_name=self.task_name,
                                            xpos_targets=self.xpos_targets,
                                            env_kwargs=self.env_kwargs,
                                            discount=self.cfg.discount,
                                            frame_stack=self.cfg.frame_stack,
                                            seed=self.cfg.seed)
        self.eval_env = make_robosuite_env(
                                           task_name=self.task_name,
                                            xpos_targets=self.xpos_targets,
                                           env_kwargs=self.env_kwargs,
                                           discount=self.cfg.discount,
                                           frame_stack=self.cfg.frame_stack,
                                           seed=self.cfg.seed+1)
        # create replay buffer
        self.data_specs = (
                      specs.Array(shape=self.train_env.obs_shape, dtype=np.uint8, name='observation'),
                      specs.Array(shape=self.train_env.body_shape, dtype=np.float32, name='body'),
                      specs.Array(shape=self.train_env.action_shape, dtype=np.float32, name='action'),
                      specs.Array(shape=(1,), dtype=np.float32, name='reward'),
                      specs.Array(shape=(1,), dtype=np.float32, name='discount'))

        self.replay_storage = ReplayBufferStorage(self.data_specs,
                                                  self.work_dir / 'buffer')
        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
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

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
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
        if self.cfg.dataset_path != '':
            pass
            #demo_episodes, demo_steps = self.load_dataset()
            #self._global_episode += demo_episodes
            #self._global_step += demo_steps
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
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
                self.train_video_recorder.init(time_step.observation)
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
                action = self.agent.act(time_step.observation,
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
            self.train_video_recorder.record(time_step.observation)
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


    def load_dataset(self):
        f = h5py.File(self.cfg.dataset_path, "r")
        demos = list(f["data"].keys())
        print('found {} trajectories'.format(len(demos)))
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        episodes = 0
        steps = 0
        all_time_steps = []
        for ind in range(len(demos)):
            ep = demos[ind]
            time_steps, total_reward = self.playback_trajectory(f['data/{}'.format(ep)])
            print('loaded demo {} with total reward {} and len {}'.format(ep, total_reward, len(time_steps)))
            # add to replay buffer each step in traj
            [self.replay_storage.add(ts) for ts in time_steps]
            all_time_steps.extend(time_steps)
            steps += len(time_steps)
            episodes += 1

            if ind == 0:
                tsr = self.eval_env.reset()
                self.video_recorder.init(self.eval_env, enabled=1)
                for time_step in  time_steps:
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                    ts = self.eval_env.step(time_step.action)
                    self.video_recorder.record(self.eval_env)

                self.video_recorder.save(f'{self.global_frame}.mp4')
        return episodes, steps

    def playback_trajectory(self, traj_grp):
        actions = np.array(traj_grp['actions'], dtype=np.float32)
        rewards = np.array(traj_grp['rewards'], dtype=np.float32)
        image_name = self.env_kwargs['camera_names']+'_image'
        traj_len = actions.shape[0]
        frames = deque([], maxlen=self.cfg.frame_stack)
        for i in range(self.cfg.frame_stack):
            first_img_dict = {}
            first_img_dict[image_name] = traj_grp['obs/{}'.format(image_name)][0]
            frames.append(self.train_env._get_image_obs(first_img_dict))
        first_action = np.zeros_like(self.train_env.action_spec[0]).astype(np.float32)
        first_obs = np.concatenate(list(frames), axis=0)
        first_time_step = ExtendedTimeStep(step_type=dm_env.StepType.FIRST,
                                        discount=self.cfg.discount,
                                        reward=0.0, observation=first_obs,
                                        action=first_action)
        time_steps = [first_time_step]
        total_reward = 0
        for i in range(traj_len):
            img_dict = {}
            img_dict[image_name] = traj_grp['obs/{}'.format(image_name)][i]
            frames.append(self.train_env._get_image_obs(img_dict))
            obs = np.concatenate(list(frames), axis=0)
            if i == traj_len-1:
                step_type = dm_env.StepType.LAST
            else:
                step_type = dm_env.StepType.MID
            time_step = ExtendedTimeStep(step_type=step_type,
                                        discount=self.cfg.discount,
                                        reward=rewards[i], observation=obs,
                                        action=actions[i])
            time_steps.append(time_step)
            total_reward += rewards[i]
        return time_steps, total_reward

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
