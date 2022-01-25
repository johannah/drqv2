# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from robosuite.utils.dh_parameters import robotDH
from IPython import embed
import utils

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, img_shape, state_shape):
        super().__init__()

        assert len(img_shape) == 3
        self.use_state_obs = False
        self.use_image_obs = False

        self.repr_dim = 0
        if img_shape[0] > 0:
            print('making an image encoder')
            self.use_image_obs = True
            self.convnet = nn.Sequential(nn.Conv2d(img_shape[0], 32, 3, stride=2),
                                         nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                         nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                         nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                         nn.ReLU())
            self.repr_dim += 32 * 35 * 35
            print('repr_dim is now', self.repr_dim)
        if state_shape[0] > 0:
            print('making a state encoder')
            self.use_state_obs = True
            out_size = 256
            self.repr_dim += out_size
            self.mlp = nn.Sequential(nn.Linear(state_shape[0], out_size),
                          nn.ReLU(), nn.Linear(out_size, out_size),
                          nn.ReLU())
            print('repr_dim is now', self.repr_dim)

        self.apply(utils.weight_init)

    def get_image(self, img_obs):
        img_obs = img_obs / 255.0 - 0.5
        h = self.convnet(img_obs)
        h = h.view(h.shape[0], -1)
        return h

    def get_state(self, state_obs):
        h = self.mlp(state_obs)
        h = h.view(h.shape[0], -1)
        return h

    def forward(self, img_obs, state_obs):
        if self.use_image_obs and self.use_state_obs:
            return torch.cat([self.get_image(img_obs), self.get_state(state_obs)], dim=-1)
        if self.use_image_obs:
            return self.get_image(img_obs)
        if self.use_state_obs:
            return self.get_state(state_obs)

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std, low=-1, high=1)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, kinematic_type, robot_dh, joint_indexes):
        super().__init__()

        self.joint_indexes = joint_indexes
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())


        self.robot_dh = robot_dh
        self.n_joints = len(self.robot_dh.npdh['DH_a'])
        self.kinematic_type = kinematic_type
        if self.kinematic_type == 'eef':
            self.input_dim = feature_dim + action_shape[0] + 3
        elif self.kinematic_type == 'abs_rel_eef_only':
            self.input_dim = feature_dim * 2
            self.kine_projection = nn.Sequential(nn.Linear(6, feature_dim),
                                                 nn.LayerNorm(feature_dim))
        else:
            self.input_dim = feature_dim + action_shape[0]
        self.Q1 = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def kinematic_view_eef(self, joint_action, body):
        # turn relative action to abs action
        joint_position = joint_action[:, self.joint_indexes] + body[:, :self.n_joints]
        eef_rot = self.robot_dh(joint_position)
        eef_pos = eef_rot[:,:3,3]
        return eef_pos

    def kinematic_view_rel_eef(self, joint_action, body):
        # turn relative action to abs action
        joint_position = joint_action[:, self.joint_indexes] + body[:, :self.n_joints]
        next_eef_pos = self.robot_dh(joint_position)[:,:3,3]
        current_eef_pos = self.robot_dh(body[:, :self.n_joints])[:,:3,3]
        rel_eef_pos = next_eef_pos - current_eef_pos
        return torch.cat([current_eef_pos, rel_eef_pos], dim=-1)

 #   def kinematic_fn_torque(self, action, body, next_body):
 #      # turn relative action to abs action
 #      desired_qpos = action[:,self.joint_indexes] + body[:,:self.n_joints]

 #      vel_pos_error = -joint_vel
 #      desired_torque = np.multiply(np.array(position_error), np.array(self.kp)) + np.multiply(vel_pos_error, self.kd)
 #      # Return desired torques plus gravity compensations
 #      self.torques = np.dot(self.mass_matrix, desired_torque) + self.torque_compensation

 #      eef_rot = self.robot_dh.torch_angle2ee(next_body[:,:self.n_joints])
 #      eef_pos = eef_rot[:,:3,3]
 #      target_pos = next_body[:,self.n_joints:self.n_joints+3]
 #      return eef_pos, target_pos

    def forward(self, obs, action, body):
        h = self.trunk(obs)
        if self.kinematic_type == 'eef':
            kine = self.kinematic_view_eef(action, body)
            h_action = torch.cat([h, action, kine], dim=-1)
        elif self.kinematic_type == 'abs_rel_eef_only':
            kine = self.kine_projection(self.kinematic_view_rel_eef(action, body))
            h_action = torch.cat([h, kine], dim=-1)
        elif self.kinematic_type == "None":
            h_action = torch.cat([h, action], dim=-1)
        else:
            raise ValueError("unknown kinematic type: %s"%self.kinematic_type)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2


class DrQV2Agent:
    def __init__(self, img_shape, state_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip,
                 use_tb, max_action, joint_indexes, robot_name, kinematic_type="None"):
        self.joint_indexes = joint_indexes
        self.max_action = torch.Tensor(max_action).to(device)
        self.robot_dh = robotDH(robot_name=robot_name, device=device)
        self.kinematic_type = kinematic_type
        self.device = device
        self.n_joints = len(self.robot_dh.npdh['DH_a'])
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        print('make an encoder')
        # models
        self.encoder = Encoder(img_shape, state_shape).to(device)
        print('finish encoder')
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim, self.max_action).to(device)

        print('finish actor')
        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, self.kinematic_type, self.robot_dh, self.joint_indexes).to(device)
        print('finish critic')
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim, self.kinematic_type, self.robot_dh, self.joint_indexes).to(device)
        print('finish critic target')
        self.critic_target.load_state_dict(self.critic.state_dict())
        print('made critic')

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        print('made optimizers')
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()
        print(' finish agent')

    def kinematic_fn(self, joint_action, body, next_body):
        # turn relative action to abs action
        joint_position = joint_action[:, :self.n_joints] + body[:, :self.n_joints]
        eef_rot = self.robot_dh(joint_position)
        eef_pos = eef_rot[:,:3,3]
        target_pos = next_body[:,self.n_joints:self.n_joints+3]
        return eef_pos, target_pos

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, img_obs, state_obs, step, eval_mode):
        if len(img_obs):
            img_obs = torch.as_tensor(img_obs, device=self.device).unsqueeze(0)
        if len(state_obs):
            state_obs = torch.as_tensor(state_obs, device=self.device).unsqueeze(0)
        obs = self.encoder(img_obs, state_obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        # WTF
        # scaled_action = (((action+1)*(2*self.max_action))/2.0)+-self.max_action
        # scale -1 to 1 bt -max to max actions
        scaled_action = action * self.max_action
        for xc, mm in enumerate(self.max_action):
            if torch.abs(scaled_action[:,xc]).max() > mm:
                print('update')
                embed()
        scaled_action =  scaled_action.cpu().numpy()[0]
        return scaled_action

    def update_critic(self, obs, body, action, reward, discount, next_obs, next_body, step):
        metrics = dict()
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            unscaled_next_action = dist.sample(clip=self.stddev_clip)
            # scale -1 to 1 to -max to max action
            next_action = unscaled_next_action * self.max_action
            #next_action = (((unscaled_next_action+1)*(2*self.max_action))/2.0)+-self.max_action
            for xc, mm in enumerate(self.max_action):
                if torch.abs(next_action[:,xc]).max() > mm:
                    print('update critic')
                    embed()

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, next_body)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action, body)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, body, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        unscaled_action = dist.sample(clip=self.stddev_clip)

        #action =(((unscaled_action+1)*(2*self.max_action))/2.0)+-self.max_action
        action = unscaled_action * self.max_action
        for xc, mm in enumerate(self.max_action):
            if torch.abs(action[:,xc]).max() > mm:
                print('update actor')
                embed()

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action, body)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        img, state, body, action, reward, discount, next_img, next_state, next_body = utils.to_torch(
            batch, self.device)

        # augment
        if img.shape[0]:
            img = self.aug(img.float())
            next_img = self.aug(next_img.float())
        # encode
        obs = self.encoder(img, state)
        with torch.no_grad():
            next_obs = self.encoder(next_img, next_state)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, body, action, reward, discount, next_obs, next_body, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), body, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
