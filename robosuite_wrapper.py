"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""
from copy import deepcopy
from typing import Any, NamedTuple
from dmc import ExtendedTimeStep
import numpy as np
from gym import spaces
from gym.core import Env
from collections import deque
import numpy as np
import dm_env
from dm_env import StepType, specs
from robosuite.wrappers import Wrapper
from robosuite.utils.mjmod import TextureModder, LightingModder, CameraModder, DynamicsModder
import robosuite.utils.transform_utils as T
from IPython import embed

DEFAULT_COLOR_ARGS = {
    'geom_names': None,  # all geoms are randomized
    'randomize_local': True,  # sample nearby colors
    'randomize_material':
    True,  # randomize material reflectance / shininess / specular
    'local_rgb_interpolation': 0.15,
    'local_material_interpolation': 0.15,
    'texture_variations': ['rgb', 'checker', 'noise',
                           'gradient'],  # all texture variation types
    'randomize_skybox': True,  # by default, randomize skybox too
}

DEFAULT_CAMERA_ARGS = {
    'camera_names': None,  # all cameras are randomized
    'randomize_position': True,
    'randomize_rotation': True,
    'randomize_fovy': True,
    'position_perturbation_size': 0.03,
    'rotation_perturbation_size': 0.02,
    'fovy_perturbation_size': 3.,
}

DEFAULT_LIGHTING_ARGS = {
    'light_names': None,  # all lights are randomized
    'randomize_position': True,
    'randomize_direction': True,
    'randomize_specular': True,
    'randomize_ambient': True,
    'randomize_diffuse': True,
    'randomize_active': True,
    'position_perturbation_size': 0.3,
    'direction_perturbation_size': 0.35,
    'specular_perturbation_size': 0.2,
    'ambient_perturbation_size': 0.3,
    'diffuse_perturbation_size': 0.3,
}

DEFAULT_DYNAMICS_ARGS = {
    # Opt parameters
    'randomize_density': True,
    'randomize_viscosity': True,
    'density_perturbation_ratio': 0.001,
    'viscosity_perturbation_ratio': 0.001,

    # Body parameters
    'body_names': None,  # all bodies randomized
    'randomize_position': True,
    'randomize_quaternion': True,
    'randomize_inertia': True,
    'randomize_mass': True,
    'position_perturbation_size': 0.00015,
    'quaternion_perturbation_size': 0.0003,
    'inertia_perturbation_ratio': 0.0002,
    'mass_perturbation_ratio': 0.0002,

    # Geom parameters
    'geom_names': None,  # all geoms randomized
    'randomize_friction': True,
    'randomize_solref': True,
    'randomize_solimp': True,
    'friction_perturbation_ratio': 0.001,
    'solref_perturbation_ratio': 0.001,
    'solimp_perturbation_ratio': 0.001,

    # Joint parameters
    'joint_names': None,  # all joints randomized
    'randomize_stiffness': True,
    'randomize_frictionloss': True,
    'randomize_damping': True,
    'randomize_armature': True,
    'stiffness_perturbation_ratio': 0.001,
    'frictionloss_perturbation_size': 0.0005,
    'damping_perturbation_size': 0.0001,
    'armature_perturbation_size': 0.0001,
}

class DRQWrapper(Wrapper):
    """
    Wrapper that allows for domain randomization mid-simulation.
    Args:
        env (MujocoEnv): The environment to wrap.
        xpos_targets: list of site names to use as EEF targets
        seed (int): Integer used to seed all randomizations from this wrapper. It is
            used to create a np.random.RandomState instance to make sure samples here
            are isolated from sampling occurring elsewhere in the code. If not provided,
            will default to using global random state.
        randomize_color (bool): if True, randomize geom colors and texture colors
        randomize_camera (bool): if True, randomize camera locations and parameters
        randomize_lighting (bool): if True, randomize light locations and properties
        randomize_dyanmics (bool): if True, randomize dynamics parameters
        color_randomization_args (dict): Color-specific randomization arguments
        camera_randomization_args (dict): Camera-specific randomization arguments
        lighting_randomization_args (dict): Lighting-specific randomization arguments
        dynamics_randomization_args (dict): Dyanmics-specific randomization arguments
        randomize_on_reset (bool): if True, randomize on every call to @reset. This, in
            conjunction with setting @randomize_every_n_steps to 0, is useful to
            generate a new domain per episode.
        randomize_every_n_steps (int): determines how often randomization should occur. Set
            to 0 if randomization should happen manually (by calling @randomize_domain)
        randomize_on_init (bool): if True: randomize on initialization and use those initial defaults for remaining randomization
    """
    def __init__(
        self,
        env,
        use_proprio_obs=False,
        seed=112,
        randomize_color=False,
        randomize_camera=False,
        randomize_lighting=False,
        randomize_dynamics=False,
        color_randomization_args=DEFAULT_COLOR_ARGS,
        camera_randomization_args=DEFAULT_CAMERA_ARGS,
        lighting_randomization_args=DEFAULT_LIGHTING_ARGS,
        dynamics_randomization_args=DEFAULT_DYNAMICS_ARGS,
        randomize_on_reset=False,
        randomize_on_init=False,
        randomize_every_n_steps=0,
        frame_stack=3,
        discount=.99,
        randomize_episode_prob=1,
    ):
        super().__init__(env)

        #assert env.use_camera_obs == True
        self.use_proprio_obs = use_proprio_obs
        self._k = frame_stack
        self._frames = deque([], maxlen=self._k)
        self.seed_value = seed
        self.discount = discount
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = None
        self.randomize_episode_prob = randomize_episode_prob
        if randomize_camera:
            camera_randomization_args['camera_names'] = env.camera_names
        self.randomize_color = randomize_color
        self.randomize_camera = randomize_camera
        self.randomize_lighting = randomize_lighting
        self.randomize_dynamics = randomize_dynamics
        self.color_randomization_args = color_randomization_args
        self.camera_randomization_args = camera_randomization_args
        self.lighting_randomization_args = lighting_randomization_args
        self.dynamics_randomization_args = dynamics_randomization_args
        self.randomize_on_reset = randomize_on_reset
        self.randomize_on_init = randomize_on_init
        self.randomize_every_n_steps = randomize_every_n_steps
        self.step_counter = 0
        self.modders = []


        # n_joints
       # damping_ratio, kp, action
        self.n_joints = len(self.env.robots[0].controller.qpos_index)
        try:
            if self.env.robots[0].controller.impedance_mode == 'fixed':
                self.joint_indexes = np.arange(self.n_joints).astype(np.int)
            elif self.env.robots[0].controller.impedance_mode == 'variable':
                self.joint_indexes = np.arange(self.n_joints*2, self.n_joints*3).astype(np.int)
            elif self.env.robots[0].controller.impedance_mode == 'variable':
                self.joint_indexes = np.arange(self.n_joints, self.n_joints*2).astype(np.int)
        except:
            self.joint_indexes = np.arange(self.n_joints).astype(np.int)

        print('environment', self.randomize_color)
        if self.randomize_color:
            if color_randomization_args['geom_names'] == None:
                use_geoms = []
                # nonexhaustive list of what not to randomize
                exclude_geoms = ['robot','gripper','mount','collision','ball', 'cube']
                for g in list(env.sim.model.geom_names):
                    exclude = [e for e in exclude_geoms if e in g]
                    if not len(exclude):
                        use_geoms.append(g)
                color_randomization_args['geom_names'] = use_geoms

            print('randomizing color geoms', color_randomization_args['geom_names'])
            self.tex_modder = TextureModder(sim=self.env.sim,
                                            random_state=self.random_state,
                                            **self.color_randomization_args)
            self.modders.append(self.tex_modder)

        if self.randomize_camera:
            self.camera_modder = CameraModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.camera_randomization_args,
            )
            self.modders.append(self.camera_modder)

        if self.randomize_lighting:
            self.light_modder = LightingModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.lighting_randomization_args,
            )
            self.modders.append(self.light_modder)

        if self.randomize_dynamics:
            self.dynamics_modder = DynamicsModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.dynamics_randomization_args,
            )
            self.modders.append(self.dynamics_modder)

        self.save_default_domain()
        self.keys = [f"{cam_name}_image" for cam_name in self.env.camera_names]

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        self.state_keys = []
        # set up observation and action spaces
        self.img_shape = (0,0,0)
        self.img_dtype = np.uint8
        state_shape = 0
        self.state_dtype = np.float32

        if self.env.use_camera_obs:
            self.img_shape = (3*self._k, self.env.camera_heights[0], self.env.camera_widths[0])
        if self.use_proprio_obs:
             for idx in range(len(self.env.robots)):
                 self.state_keys += ['robot{}_proprio-state'.format(idx)]

        if self.env.use_object_obs:
             # we'll need to flatten the observations
             robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
             self.state_keys.append('object-state')
        for key in self.state_keys:
            state_shape += self.env.observation_spec()[key].shape[0]
        self.state_shape = (state_shape,)
        low, high = self.env.action_spec
        self.max_actions = high
        self.action_shape = low.shape
        self.action_dtype = np.float32
        self.body_shape = (len(self.make_body()),)
        self.body_dtype = np.float32
        # This is for Jaco
        self.base_matrix = np.array([[0,1,0,0],
                                     [1,0,0,0],
                                     [0,0,-1,0],
                                     [0,0,0,1]])
        self.bpos = self.base_matrix[:3, 3]
        self.bori = T.mat2quat(self.base_matrix)

        self.controller_iterations = int(self.env.control_timestep / self.env.model_timestep)
        self._max_episode_steps = self.env.horizon
        if self.randomize_on_init:
            print('setting initial randomization')
            self.randomize_domain()
            # save new default
            self.save_default_domain()

    def site_pose_in_base_from_name(self, root_body, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
        Args:
            root_body (str): reference pose (usually robot base pose)
            name (str): Name of site in sim to grab pose
        Returns:
            np.array: (4,4) array corresponding to the pose of @name in the base frame
        """

        pos_in_world = self.env.sim.data.get_site_xpos(name)
        rot_in_world = self.env.sim.data.get_site_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.env.sim.data.get_body_xpos(root_body)
        base_rot_in_world = self.env.sim.data.get_body_xmat(root_body).reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def make_body(self):
        # TODO only works with single robot
        r =  self.env.robots[0]
        # include joint qpos and qvel
        bxqs = np.hstack((deepcopy(r._joint_positions), deepcopy(r._joint_velocities)))
        return bxqs.astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward
        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]
        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

    def seed(self, seed=None):
        """
        Utility function to set numpy seed
        Args:
            seed (None or int): If specified, numpy seed to set
        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def _get_image_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.
        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed
        Returns:
            np.array: image observations into an array combined across channels
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                obs_pix = obs_dict[key][::-1]
                obs_pix = obs_pix.swapaxes(0,2)
                ob_lst.append(np.array(obs_pix))
        # concatenate over channels
        if len(ob_lst):
            ob_lst = np.concatenate(ob_lst, 2)
        return ob_lst

    def reset(self):
        """
        Extends superclass method to reset the domain randomizer.
        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        # undo all randomizations
        self.restore_default_domain()

        # normal env reset
        ret = super().reset()

        # save the original env parameters
        self.save_default_domain()

        # reset counter for doing domain randomization at a particular frequency
        self.step_counter = 0

        # update sims
        for modder in self.modders:
            modder.update_sim(self.env.sim)

        if self.randomize_on_reset:
            # domain randomize + regenerate observation
            self.randomize_domain()
            ret = self.env._get_observations()

        img = self._get_image_obs(ret)
        state = self.make_state(ret)
        b = self.make_body()
        for _ in range(self._k):
            self._frames.append(img)
        action = np.zeros_like(self.action_spec[0]).astype(np.float32)
        return ExtendedTimeStep(step_type=dm_env.StepType.FIRST,
                                discount=self.discount,
                                reward=0.0,
                                img_obs=self._get_stack_obs(),
                                state_obs=state,
                                body=b,
                                action=action)

    def make_state(self, obs_dict):
        obs = []
        for key in self.state_keys:
            obs.extend(obs_dict[key])
        return np.array(obs).astype(np.float32)

    def _get_stack_obs(self):
        assert len(self._frames) == self._k
        concat_frames = np.concatenate(list(self._frames), axis=0)
        return concat_frames

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate domain randomization
        Returns:
            4-tuple:
                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """

        #assert np.abs(action).max() <= 0.0150
        # functionality for randomizing at a particular frequency
        if self.randomize_every_n_steps > 0:
            if self.step_counter % self.randomize_every_n_steps == 0:
                self.randomize_domain()
        self.step_counter += 1

        ob_dict, reward, done, info = self.env.step(action)
        #reward += -np.linalg.norm(action[self.joint_indexes])/self.n_joints
        self._frames.append(self._get_image_obs(ob_dict))
        img = self._get_stack_obs()
        body = self.make_body()
        state = self.make_state(ob_dict)
        #return obs, reward, done, info
        if not done:
            step_type = dm_env.StepType.MID
        else:
            step_type = dm_env.StepType.LAST
        return ExtendedTimeStep(step_type=step_type,
                                discount=self.discount,
                                reward=reward,
                                img_obs=img,
                                state_obs=state,
                                body=body,
                                action=action)

    def randomize_domain(self):
        """
        Runs domain randomization over the environment.
        """
        if len(self.modders):
            print('========updating randomization')
        for modder in self.modders:
            if self.random_state.rand() < self.randomize_episode_prob:
                 modder.randomize()

    def save_default_domain(self):
        """
        Saves the current simulation model parameters so
        that they can be restored later.
        """
        for modder in self.modders:
            modder.save_defaults()

    def restore_default_domain(self):
        """
        Restores the simulation model parameters saved
        in the last call to @save_default_domain.
        """
        for modder in self.modders:
            modder.restore_defaults()

    def render(self,
               mode='human',
               width=256,
               height=256,
               depth=False,
               task_name=None):
        """ render based on ibit call. mode / task_name are unused"""
        data = self.env.sim.render(camera_name=self.env.camera_names[0],
                                   width=width,
                                   height=height,
                                   depth=depth)
        # original image is upside-down and mirrored, so flip both axis
        if not depth:
            return data[::-1]
        else:
            # Untested
            return data[0][::-1, :], data[1][::-1]
