# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from robosuite.utils.dh_parameters import robot_attributes
from IPython import embed
import utils


def torch_dh_transform(theta, d, a, alpha, device):
    bs = theta.shape[0]
    T = torch.zeros((bs,4,4), device=device)
    T[:,0,0] = T[:,0,0] +  torch.cos(theta)
    T[:,0,1] = T[:,0,1] + -torch.sin(theta)*torch.cos(alpha)
    T[:,0,2] = T[:,0,2] +  torch.sin(theta)*torch.sin(alpha)
    T[:,0,3] = T[:,0,3] +  a*torch.cos(theta)
    T[:,1,0] = T[:,1,0] +  torch.sin(theta)
    T[:,1,1] = T[:,1,1] +   torch.cos(theta)*torch.cos(alpha)
    T[:,1,2] = T[:,1,2] +   -torch.cos(theta)*torch.sin(alpha)
    T[:,1,3] = T[:,1,3] +  a*torch.sin(theta)
    T[:,2,1] = T[:,2,1] +  torch.sin(alpha)
    T[:,2,2] = T[:,2,2] +   torch.cos(alpha)
    T[:,2,3] = T[:,2,3] +  d
    T[:,3,3] = T[:,3,3] +  1.0
    return T

def np_dh_transform(theta, d, a, alpha):
    bs = theta.shape[0]
    T = np.zeros((bs,4,4))
    T[:,0,0] = T[:,0,0] +  np.cos(theta)
    T[:,0,1] = T[:,0,1] + -np.sin(theta)*np.cos(alpha)
    T[:,0,2] = T[:,0,2] +  np.sin(theta)*np.sin(alpha)
    T[:,0,3] = T[:,0,3] +  a*np.cos(theta)
    T[:,1,0] = T[:,1,0] +    np.sin(theta)
    T[:,1,1] = T[:,1,1] +    np.cos(theta)*np.cos(alpha)
    T[:,1,2] = T[:,1,2] +   -np.cos(theta)*np.sin(alpha)
    T[:,1,3] = T[:,1,3] +  a*np.sin(theta)
    T[:,2,1] = T[:,2,1] +  np.sin(alpha)
    T[:,2,2] = T[:,2,2] +   np.cos(alpha)
    T[:,2,3] = T[:,2,3] +  d
    T[:,3,3] = T[:,3,3] +  1.0
    return T


class robotDH():
    def __init__(self, robot_name, device='cpu'):
        self.device = device
        self.robot_name = robot_name
        self.npdh = robot_attributes[self.robot_name]
        self.base_matrix = robot_attributes[self.robot_name]['base_matrix']
        self.t_base_matrix = torch.Tensor(robot_attributes[self.robot_name]['base_matrix']).to(self.device)
        self.tdh = {}
        for key, item in self.npdh.items():
            self.tdh[key] = torch.FloatTensor(item).to(self.device)

    def np_angle2ee(self, angles):
        """
            convert np joint angle to end effector for for ts,angles (in radians)
        """
        # ts, bs, feat
        ts, fs = angles.shape
        _T = self.base_matrix
        for _a in range(fs):
            _T1 = self.np_dh_transform(_a, angles[:,_a])
            _T = np.matmul(_T, _T1)
        return _T

    def torch_angle2ee(self, angles):
        """
            convert joint angle to end effector for reacher for ts,bs,f
        """
        # ts, bs, feat
        ts, fs = angles.shape
        #ee_pred = torch.zeros((ts,4,4)).to(self.device)
        _T = self.t_base_matrix
        for _a in range(fs):
            _T1 = self.torch_dh_transform(_a, angles[:,_a])
            _T = torch.matmul(_T, _T1)
        return _T

    def np_dh_transform(self, dh_index, angles):
        theta = self.npdh['DH_theta_sign'][dh_index]*angles+self.npdh['DH_theta_offset'][dh_index]
        d = self.npdh['DH_d'][dh_index]
        a = self.npdh['DH_a'][dh_index]
        alpha = self.npdh['DH_alpha'][dh_index]
        return np_dh_transform(theta, d, a, alpha)

    def torch_dh_transform(self, dh_index, angles):
        theta = self.tdh['DH_theta_sign'][dh_index]*angles+self.tdh['DH_theta_offset'][dh_index]
        d = self.tdh['DH_d'][dh_index]
        a = self.tdh['DH_a'][dh_index]
        alpha = self.tdh['DH_alpha'][dh_index]
        return torch_dh_transform(theta, d, a, alpha, self.device)


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
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


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
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip,
                 use_tb, max_action, robot_name="Jaco", use_kinematic_loss=False, kine_weight=1):
        self.kine_weight = kine_weight
        self.max_action = torch.Tensor(max_action).to(device)
        self.robot_name = robot_name
        self.use_kinematic_loss = use_kinematic_loss
        if self.use_kinematic_loss:
            print("using kinematic loss")
        assert 0 < self.max_action.max() <= 1
        self.device = device
        self.robot_dh = robotDH(robot_name=self.robot_name, device=self.device)
        self.n_joints = len(self.robot_dh.npdh['DH_a'])
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim, self.max_action).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def kinematic_fn(self, joint_action, body, next_body):
        # turn relative action to abs action
        joint_position = joint_action[:, :self.n_joints] + body[:, :self.n_joints]
        eef_rot = self.robot_dh.torch_angle2ee(joint_position)
        eef_pos = eef_rot[:,:3,3]
        target_pos = next_body[:,self.n_joints:self.n_joints+3]
        return eef_pos, target_pos

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        # scale bt min/max actions
        scaled_action = (((action+1)*(2*self.max_action))/2.0)+-self.max_action
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

            next_action = (((unscaled_next_action+1)*(2*self.max_action))/2.0)+-self.max_action
            for xc, mm in enumerate(self.max_action):
                if torch.abs(next_action[:,xc]).max() > mm:
                    print('update critic')
                    embed()

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        if self.use_kinematic_loss:
            eef_pos, target_pos = self.kinematic_fn(action, body, next_body)
            kine_loss = self.kine_weight * F.mse_loss(eef_pos, target_pos)
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q) + kine_loss
        else:
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            if self.use_kinematic_loss:
                metrics['kine_loss'] = kine_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        unscaled_action = dist.sample(clip=self.stddev_clip)

        action = (((unscaled_action+1)*(2*self.max_action))/2.0)+-self.max_action
        for xc, mm in enumerate(self.max_action):
            if torch.abs(action[:,xc]).max() > mm:
                print('update actor')
                embed()

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
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
        obs, body, action, reward, discount, next_obs, next_body = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, body, action, reward, discount, next_obs, next_body, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
