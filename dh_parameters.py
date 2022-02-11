import numpy as np
import torch
import torch.nn as nn
import robosuite.utils.transform_utils as T
"""
DH params found by Yuying Blair Huang
"""
# Params for Denavit-Hartenberg Reference Frame Layout (DH)
jaco27DOF_DH_lengths = {'D1':0.2755, 'D2':0.2050,
                        'D3':0.2050, 'D4':0.2073,
                        'D5':0.1038, 'D6':0.1038,
                        'D7':0.1600, 'e2':0.0098, 'D_grip':0.1775} # .001775e2 is dist to grip site


DH_attributes_jaco27DOF = {
          'base_matrix': np.array([[0,1,0,0],
                                   [1,0,0,0],
                                   [0,0,-1,0],
                                   [0,0,0,1]]),
          'DH_a':[0, 0, 0, 0, 0, 0, 0],
          'DH_alpha':[np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi],
          'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1],
          'DH_theta_offset':[np.pi,0.0, 0.0, 0.0, 0.0,0.0,np.pi/2.0],
          'DH_d':(-jaco27DOF_DH_lengths['D1'],
                    0,
                    -(jaco27DOF_DH_lengths['D2']+jaco27DOF_DH_lengths['D3']),
                    -jaco27DOF_DH_lengths['e2'],
                    -(jaco27DOF_DH_lengths['D4']+jaco27DOF_DH_lengths['D5']),
                    0,
                    -(jaco27DOF_DH_lengths['D6']+jaco27DOF_DH_lengths['D_grip']))
           }
#
#DH_attributes_jaco27DOF = {
#          'DH_a':[0, 0, 0, 0, 0, 0, 0],
#          'DH_alpha':[np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi],
#          'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1],
#          'DH_theta_offset':[np.pi,0.0, 0.0, 0.0, 0.0,0.0,np.pi/2.0],
#          'DH_d':(-jaco27DOF_DH_lengths['D1'],
#                    0,
#                    -(jaco27DOF_DH_lengths['D2']+jaco27DOF_DH_lengths['D3']),
#                    -jaco27DOF_DH_lengths['e2'],
#                    -(jaco27DOF_DH_lengths['D4']+jaco27DOF_DH_lengths['D5']),
#                    0,
#                    -(jaco27DOF_DH_lengths['D6']+jaco27DOF_DH_lengths['D_grip']))
#           }
# Params for Denavit-Hartenberg Reference Frame Layout (DH)





Panda_DH_lengths = {'D1':0.333,
               'D3':0.316,
               'D5':0.384,
               'DF':0.1065, "D_grip":0.097, 'e1':0.0825, 'j7':0.088}


DH_attributes_Panda = {
          'DH_a':[0, 0, 0, Panda_DH_lengths['e1'], -Panda_DH_lengths['e1'], 0, Panda_DH_lengths['j7'], 0],
           'DH_alpha':[0, -np.pi/2.0, np.pi/2.0, np.pi/2.0, -np.pi/2.0, np.pi/2.0, np.pi/2.0, 0],
           'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1, 0],
           'DH_theta_offset':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           'DH_d':(Panda_DH_lengths['D1'],
                   0.0,
                   Panda_DH_lengths['D3'],
                   0.0,
                   Panda_DH_lengths['D5'],
                   0.0,
                   0.0,
                   Panda_DH_lengths['D_grip'])
           }


Sawyer_DH_lengths = {'D1':0.237, 'D2':0.1925,
               'D3':0.4, 'D4':-0.1685,
               'D5':0.4, 'D6':0.1363,
               'D7':0.11, 'e1':0.081}


DH_attributes_Sawyer = {
          'DH_a':[Sawyer_DH_lengths['e1'], 0, 0, 0, 0, 0, 0],
           'DH_alpha':[-np.pi/2.0, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0,0],
           'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1],
           'DH_theta_offset':[np.pi,0.0, 0.0, 0.0, 0.0,0.0,np.pi/2.0],
           'DH_d':(Sawyer_DH_lengths['D1'],
                   Sawyer_DH_lengths['D2'],
                   Sawyer_DH_lengths['D3'],
                   Sawyer_DH_lengths['D4'],
                   Sawyer_DH_lengths['D5'],
                   Sawyer_DH_lengths['D6'],
                   Sawyer_DH_lengths['D7'])
           }

Baxter_DH_lengths = {'D1':0.27035, 'D2':0.102,
               'D3':0.26242, 'D4':0.10359,
               'D5':0.2707,  'D6':0.115975,
               'D7':0.11355, 'e1':0.069, 'e2':0.010}


DH_attributes_Baxter = {
          'DH_a':[Baxter_DH_lengths['e1'], 0, Baxter_DH_lengths['e1'], 0, Baxter_DH_lengths['e2'], 0, 0],
           'DH_alpha':[-np.pi/2.0, np.pi/2.0, -np.pi/2.0, np.pi/2.0, -np.pi/2.0, np.pi/2.0,0],
           'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1],
           'DH_theta_offset':[0.0, np.pi/2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           'DH_d':(Baxter_DH_lengths['D1'],
                   0,
                   Baxter_DH_lengths['D2']+Baxter_DH_lengths['D3'],
                   0,
                   Baxter_DH_lengths['D4']+Baxter_DH_lengths['D5'],
                   0,
                   Baxter_DH_lengths['D6']+Baxter_DH_lengths['D7'])
           }

DH_attributes_dm_reacher = {
    'base_matrix':np.array([[1.  , 0.  , 0.  , 0.  ],
                            [0.  , -1.  , 0.  , 0.  ],
                            [0.  , 0.  , -1.  , 0.],
                            [0.  , 0.  , 0.  , 1.  ]]),
     'DH_a':[0.12,0.12], # arm is .12, hand is .1 + .01 sphere finger
     'DH_alpha':[0.0,0.0],
     'DH_theta_sign':[1.0,1.0],
     'DH_theta_offset':[0,0],
     'DH_d':[0,0]}

DH_attributes_dm_reacher_long_wrist = {
     'DH_a':[0.12,0.22],
     'DH_alpha':[0.0,0.0],
     'DH_theta_sign':[1.0,1.0],
     'DH_theta_offset':[0,0],
     'DH_d':[0,0]}

DH_attributes_dm_reacher_double = {
     'DH_a':[0.22,0.22],
     'DH_alpha':[0.0,0.0],
     'DH_theta_sign':[1.0,1.0],
     'DH_theta_offset':[0,0],
     'DH_d':[0,0]}




robot_attributes = {
                    'reacher':DH_attributes_dm_reacher,
                    'reacher_long_wrist':DH_attributes_dm_reacher_long_wrist,
                    'reacher_double':DH_attributes_dm_reacher_double,
                    'Jaco':DH_attributes_jaco27DOF,
                    'Baxter':DH_attributes_Baxter,
                    'Sawyer':DH_attributes_Sawyer,
                    'Panda':DH_attributes_Panda,
                   }
def build_torch_dh_matrix(theta, d, a, alpha, device):
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


class robotDH(nn.Module):
    def __init__(self, robot_name, device):
        super(robotDH, self).__init__()
        self.robot_name = robot_name
        self.device = device
        self.npdh = robot_attributes[self.robot_name]
        self.base_matrix = robot_attributes[self.robot_name]['base_matrix']
        self.t_base_matrix = torch.Tensor(robot_attributes[self.robot_name]['base_matrix']).to(device)
        self.tdh = {}
        for key, item in self.npdh.items():
            self.tdh[key] = torch.FloatTensor(item).to(device)

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

    def forward(self, angles):
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
        return build_torch_dh_matrix(theta, d, a, alpha, angles.device)

def dm_site_pose_in_base_from_name(physics, root, name):
    """
    A helper function that takes in a named data field and returns the pose
    of that object in the base frame.
    Args:
        name (str): Name of site in sim to grab pose
    Returns:
        np.array: (4,4) array corresponding to the pose of @name in the base frame
    """
    pos_in_world = physics.named.data.geom_xpos[name]
    rot_in_world = physics.named.data.geom_xmat[name].reshape((3, 3))
    pose_in_world = T.make_pose(pos_in_world, rot_in_world)

    base_pos_in_world =  physics.named.data.geom_xpos[root]
    base_rot_in_world =  physics.named.data.geom_xmat[root].reshape((3, 3))
    base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
    world_pose_in_base = T.pose_inv(base_pose_in_world)

    pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
    #from IPython import embed; embed()
    return pose_in_base
