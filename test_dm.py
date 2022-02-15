import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
np.set_printoptions(suppress=True)
import dmc
import imageio
from dh_parameters import robotDH, dm_site_pose_in_base_from_name
import torch
import robosuite.utils.transform_utils as T
from IPython import embed;
from dm_control import suite
device = 'cpu'
#for domain_name, task_name in suite.BENCHMARKING:
#    print(domain_name, task_name)
task_name = 'manipulator_bring_ball'
#task_name = 'reacher_easy'
frame_stack = 3
action_repeat = 2
seed = 3
frames = []
env = dmc.make_dm(task_name, frame_stack, action_repeat, seed, use_joint_position=False)
robot_dh = robotDH(env.robot_name, device=device)
action_dim = env.action_spec().shape[0]
max_actions = env.action_spec().maximum

pred_pos = []
true_pos  = []
pred_quat = []
true_quat  = []
for e in range(1):
    obs = env.reset()
    target_pos = env.physics.named.data.geom_xpos[robot_dh.keys['target'][0]]
    eef_pos =  env.physics.named.data.geom_xpos[robot_dh.keys['eef'][0]]
    while not obs.last():
        eef_err = target_pos-eef_pos
        action = np.zeros(action_dim)
        action[0] = .1
        obs = env.step(action)
        img = obs.img_obs[(frame_stack-1)*3:(frame_stack)*3].swapaxes(0,2)
        qpos =  env.physics.named.data.qpos[robot_dh.keys['joint_names']]
        next_eef = robot_dh(torch.FloatTensor(qpos)[None,:]).cpu().detach().numpy()
        pred_eef_pos = next_eef[0,:3,3]
        pred_eef_quat = T.mat2quat(next_eef[0])
        eef_pose = deepcopy(dm_site_pose_in_base_from_name(env.physics, robot_dh.keys['root'], robot_dh.keys['eef'][0]))
        base_eef_pos = eef_pose[:3,3]
        base_eef_quat =  T.mat2quat(eef_pose)
        pred_pos.append(deepcopy(pred_eef_pos))
        true_pos.append(deepcopy(base_eef_pos))
        pred_quat.append(deepcopy(pred_eef_quat))
        true_quat.append(deepcopy(base_eef_quat))
        frames.append(img)
pred_pos = np.array(pred_pos)
true_pos = np.array(true_pos)
vmin = min([pred_pos.min(), true_pos.min()])
vmax = max([pred_pos.max(), true_pos.max()])
f, ax = plt.subplots(3)
for ii, idx in enumerate(['x', 'y', 'z']):
    ax[ii].plot(np.array(pred_pos)[:,ii], label='pred %s'%idx)
    ax[ii].plot(np.array(true_pos)[:,ii], label='true %s'%idx, linestyle=':', linewidth=4)
    ax[ii].set_ylim(vmin, vmax)
plt.legend()
plt.savefig(task_name+'_xyz.png')
plt.close()

pred_quat = np.array(pred_quat)
true_quat = np.array(true_quat)
vmin = min([pred_quat.min(), true_quat.min()])
vmax = max([pred_quat.max(), true_quat.max()])
f, ax = plt.subplots(4)
for ii, idx in enumerate(['qx', 'qy', 'qz', 'qw']):
    ax[ii].plot(np.array(pred_quat)[:,ii], label='pred %s'%idx)
    ax[ii].plot(np.array(true_quat)[:,ii], label='true %s'%idx, linestyle=':', linewidth=4)
    ax[ii].set_ylim(vmin, vmax)
plt.legend()
plt.savefig(task_name+'_quat.png')

imageio.mimsave(task_name+'.mp4', frames)
