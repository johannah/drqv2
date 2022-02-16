# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from IPython import embed
from dh_parameters import robotDH

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    from: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))



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
    def __init__(self, repr_dim, action_shape, body_dim, feature_dim, hidden_dim):
        super().__init__()

        self.body_dim = body_dim
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
        dist = utils.TruncatedNormal(mu, std)
        return dist

#class Controller(nn.Module):
#    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, joint_indexes, robot_dh, kine_type='None', controller_iterations=1):
#        super().__init__()
#
#        self.controller_iterations = control_iterations
#        self.robot_dh = robot_dh
#        self.joint_indexes = joint_indexes
#        self.n_joints = len(self.joint_indexes)
#        self.kine_type = kine_type
#        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
#                                   nn.LayerNorm(feature_dim), nn.Tanh())
#        self.input_size = feature_dim + action_shape[0]


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, joint_indexes, robot_dh, kine_type='None', controller_iterations=1):
        super().__init__()

        self.controller_iterations = controller_iterations
        self.robot_dh = robot_dh
        self.joint_indexes = joint_indexes
        self.n_joints = len(self.joint_indexes)
        self.kine_type = kine_type
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.input_size = feature_dim + action_shape[0]

        self.eef_type = 'None'
        self.use_body = False
        if 'body' in self.kine_type:
            self.use_body = True
            self.input_size += len(self.joint_indexes)
        self.dh_size = 0
        self.return_pose = False
        self.return_quat = False
        if '_posquat' in self.kine_type:
            self.dh_size = 7
            self.eef_type = 'posquat'
        elif '_pos' in self.kine_type:
            self.dh_size = 3
            self.eef_type = 'pos'
        elif '_quat' in self.kine_type:
            self.dh_size = 4
            self.eef_type = 'quat'
        else:
            self.dh_size = 16
            self.eef_type = 'mat'
        print('size of dh inputs', self.dh_size)

        self.num_dh = 0
        # are we including relative and abs eef?
        # TODO this isn't coherent yet
        self.abs_eef = False
        self.rel_eef = False
        if 'DH' in self.kine_type:
            if 'rel' in self.kine_type:
                self.num_dh += 1
                self.rel_eef = True
            if 'abs' in self.kine_type:
                self.abs_eef = True
                self.num_dh += 1
            # fall back to abs eef on
            if not (self.abs_eef + self.rel_eef):
                self.abs_eef = True
                self.num_dh += 1
        print('number of dh inputs', self.num_dh)
        self.input_size += self.num_dh*self.dh_size
        print('using kinematic type', self.kine_type, self.input_size)
        self.Q1 = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

        self.controller_input_size = 0
        if 'controller' in self.kine_type:
            self.controller_input_size = feature_dim + self.n_joints + self.n_joints*2
            if 'structured' in self.kine_type:
                self.controller_input_size += self.n_joints
            self.inverse_controller = nn.Sequential(nn.Linear(self.controller_input_size, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, self.n_joints))
        print('using controller', self.controller_input_size)

    def run_inverse_controller(self, h, action, joint_position, joint_velocity):
        # desired_torques = torch.multiply(position_error, kp) + torch.multiply(vel_pos_error, kd)
        # torques = je*kp + jv*kd
        # (n/kp)(torques - (jv*kd)) = je
        torques = action[:,:self.n_joints]
        if self.controller_input_size == 0:
            return torques
        else:
            input_data = [h, torques, joint_position, joint_velocity]
            if 'structured' in self.kine_type:
                kp = 200; kd = 0.3
                joint_diff = (self.controller_iterations/kp) * (torques -  torch.multiply(-joint_velocity, kd))
                input_data.append(joint_diff)
            # tanh to max rel pos diff is -1, 1
            relative_joint_position = torch.tanh(self.inverse_controller(torch.cat(input_data, dim=-1)))
            return relative_joint_position

    def get_eef_rep(self, matrix):
        bs = matrix.shape[0]
        if self.eef_type == 'mat':
            return matrix.view(bs, 4*4)
        elif self.eef_type == 'pos':
            return matrix[:,:3,3]
        elif self.eef_type == 'quat':
            return matrix_to_quaternion(matrix[:,:3, :3])
        elif self.eef_type == 'posquat':
            pos =  matrix[:,:3,3]
            quat =  matrix_to_quaternion(matrix[:,:3, :3])
            return torch.cat([pos, quat], dim=-1)
        else:
            print('unable to handle')
            embed()
            raise ValueError; 'unable to handle %s'%self.eef_type

    def kinematic_view_eef(self, joint_action, joint_position):
        # turn relative action to abs action
        bs = joint_action.shape[0]
        next_joint_position = joint_action[:, self.joint_indexes] + joint_position
        # next eef position in abs position
        next_eef = self.robot_dh(next_joint_position)
        eef_return = []
        if self.abs_eef:
            eef_return.append(self.get_eef_rep(next_eef))
        if self.rel_eef:
            current_eef = self.robot_dh(joint_position)
            # next eef position relative to current position
            rel_eef = next_eef - current_eef
            eef_return.append(self.get_eef_rep(rel_eef))
        return torch.cat(eef_return, dim=-1)

#    def forward_control(self, relative_joint_qpos, joint_qpos, joint_qvel, num_iterations):
#        # num_iterations in robosuite is: int(self.control_timestep / self.model_timestep)
#        # torques = pos_err * kp + vel_err * kd
#        kp = 200
#        kd = .3
#        desired_qpos = joint_qpos + relative_joint_qpos
#        position_error = desired_qpos - joint_qpos
#        vel_pos_error = -joint_qvel
#        desired_torques = torch.multiply(position_error, kp) + torch.multiply(vel_pos_error, kd)
#        # Return desired torques plus gravity compensations
#        torques = num_iterations*(desired_torque + self.torque_compensation)
#        return torques
#
    #def forward(self, obs, action, body):
    #    control_torques = self.run_control(action[:,:self.n_joints], body[:,:self.n_joints], body[:,self.n_joints:(2*self.n_joints)], self.num_iterations)
    #    torques = self.controller(np.cat([obs, control_torques], dim=-1))
    #    return torques


    def forward(self, obs, action, body):
        h = self.trunk(obs)
        input_cats = [h, action]
        joint_position = body[:, :self.n_joints]
        joint_velocity = body[:, self.n_joints:(self.n_joints*2)]
        if self.use_body:
            input_cats.append(joint_position)
        # estimate the relative joint position, given this action
        relative_joint_position = self.run_inverse_controller(h, action, joint_position, joint_velocity)
        eef = self.kinematic_view_eef(relative_joint_position, joint_position)
        if self.abs_eef + self.rel_eef:
            input_cats.append(eef)
        h_action = torch.cat(input_cats, dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        if torch.isnan(min([q1.min(), q2.min(), eef.min()])):
            embed()
        return q1, q2, eef[:,:self.dh_size]


class DrQV2Agent:
    def __init__(self, img_shape, state_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip,
                 use_tb, max_actions, joint_indexes, robot_name, experiment_type, controller_iterations):
        self.controller_iterations = controller_iterations
        self.experiment_type = experiment_type
        self.joint_indexes = joint_indexes
        self.n_joints = len(self.joint_indexes)
        self.max_actions = torch.Tensor(max_actions).to(device)
        self.robot_name = robot_name
        self.device = device
        self.n_joints = len(self.joint_indexes)
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.body_dim = len(joint_indexes)
        self.robot_dh = robotDH(robot_name, device)
        self.kine_type = 'None'
        if 'kine' in experiment_type:
            self.kine_type = experiment_type


        # models
        self.encoder = Encoder(img_shape, state_shape).to(device)

        self.actor = Actor(self.encoder.repr_dim, action_shape, self.body_dim, feature_dim,
                           hidden_dim).to(device)


        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, self.joint_indexes,
                             robot_dh=self.robot_dh, kine_type=self.kine_type, controller_iterations=controller_iterations).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim, self.joint_indexes,
                                    robot_dh=self.robot_dh, kine_type=self.kine_type, controller_iterations=controller_iterations).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        if self.critic.controller_input_size:
            self.controller_opt = torch.optim.Adam(self.critic.inverse_controller.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

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

        scaled_action = action * self.max_actions
        scaled_action =  scaled_action.cpu().numpy()[0]
        return scaled_action

    def update_critic(self, obs, body, action, reward, discount, next_obs, next_body, step):
        metrics = dict()
        bs = obs.shape[0]
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            unscaled_next_action = dist.sample(clip=self.stddev_clip)
            # scale -1 to 1 to -max to max action
            next_action = unscaled_next_action * self.max_actions
            target_Q1, target_Q2, _ = self.critic_target(next_obs, next_action, next_body)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2, pred_next_eef = self.critic(obs, action, body)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        controller_loss = 0.0
        # train controller
        if self.critic.controller_input_size:
            no_action = torch.zeros((bs, self.n_joints)).to(self.device)
            next_eef = self.critic.kinematic_view_eef(no_action, next_body[:,:self.n_joints])
            controller_loss = F.mse_loss(pred_next_eef, next_eef)
            self.controller_opt.zero_grad(set_to_none=True)
            controller_loss.backward(retain_graph=True)

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
        if self.critic.controller_input_size:
            self.controller_opt.step()

        return metrics

    def update_actor(self, obs, body, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        unscaled_action = dist.sample(clip=self.stddev_clip)
        action = unscaled_action * self.max_actions
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2, pred_next_eef = self.critic(obs, action, body)
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
