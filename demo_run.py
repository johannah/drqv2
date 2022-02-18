import os
import imageio
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.robots import Bimanual
import numpy as np
from robosuite_wrapper import DRQWrapper
from IPython import embed
if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    num_episodes = 5
    control_freq = 10
    robot = "Jaco"
    # Choose environment and add it to options
    frame_stack = 3
    options["env_name"] = "ReachRandomStart"
    options["robots"] = robot

    controller_name = 'JOINT_POSITION'
    controller_file = "%s_%s_%shz.json" %(robot.lower(), controller_name.lower(), control_freq)
    controller_fpath = os.path.join(
                     os.path.split(suite.__file__)[0], 'controllers', 'config',
                     controller_file)

    # Load the desired controller
    options["controller_configs"] = suite.load_controller_config(custom_fpath=controller_fpath)

    # Define the number of timesteps to use per controller action as well as timesteps in between actions
    steps_per_action = 75
    steps_per_rest = 75
    camera_name = 'nearfrontview'
    # initialize the task
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=[camera_name],
        camera_widths=[256],
        camera_heights=[256],
        horizon=100,
        control_freq=control_freq,
    )
    env = DRQWrapper(env, use_proprio_obs=False, frame_stack=3,
                     randomize_on_reset=True,
                     randomize_color=True, randomize_camera=True, randomize_lighting=True,
                     randomize_dynamics=True)

    #env = DRQWrapper(env, use_proprio_obs=False, frame_stack=frame_stack,
    #                 randomize_on_reset=False,
    #                 randomize_color=False, randomize_camera=False, randomize_lighting=False,
    #                 randomize_dynamics=False, color_randomization_args=random_color_dict)
    env.reset()

    # Define neutral value
    action = np.random.rand(env.action_shape[0])

    # Keep track of done variable to know when to break loop
    count = 0
    # Loop through controller space
    video_writer = imageio.get_writer('reach_example.mp4', fps=10)
    while count < num_episodes:
        print('starting new episode')
        for i in range(10):
            action = np.random.rand(env.action_shape[0])
            o = env.step(action)
            # get last frame in frame stack
            video_writer.append_data(o.img_obs[((frame_stack-1)*3):].swapaxes(0,2))
        count += 1
        env.reset()

    # Shut down this env before starting the next test
    env.close()
    video_writer.close()
