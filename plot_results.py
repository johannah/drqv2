import matplotlib
matplotlib.use("Agg")
import os
import sys
import plotly.express as px
import pandas as pd
from glob import glob
from IPython import embed
import yaml

def load_config(yaml_path):
    loaded = {}
    with open(yaml_path, "r") as stream:
        try:
            loaded = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return loaded


frames2difficulty = {1100000:'easy',  3100000:'medium', 30100000:'hard'}
stddev2difficulty = {'linear(1.0,0.1,100000)':'easy',
                     'linear(1.0,0.1,500000)':'medium',
                     'linear(1.0,0.1,1000000)':'medium-hard',
                     'linear(1.0,0.1,2000000)':'hard',}

rolling = 100


for task in ['reacher', 'reach_', 'lift']:
    train_paths = glob(os.path.join('exp_local', '2022*', '*%s*'%task, 'train.csv'))
    train_paths = sorted(train_paths)
    print('found', len(train_paths), task)
    start = True
    for pp in train_paths:
        config_path = pp.replace('train.csv', '.hydra/config.yaml')
        config_yaml = load_config(config_path)
        n_train_frames = config_yaml['num_train_frames']
        stddev = config_yaml['stddev_schedule']
        difficulty = stddev2difficulty[stddev]
        loaded = pd.read_csv(pp)
        loaded['phase'] = 'train'
        if loaded['step'].max() > 1000:
            exp_name = os.path.split(os.path.split(pp)[0])[1]
            exp_id = exp_name[:6]
            date = os.path.split(os.path.split(os.path.split(pp)[0])[0])[1]
            day = int(date[-2:])
            eval_loaded = pd.read_csv(pp)
            eval_loaded['phase'] = 'eval'
            eval_loaded['episode_reward_smooth']  = eval_loaded['episode_reward'].rolling(rolling).mean()
            base_name = date + exp_name
            loaded['episode_reward_smooth']  = loaded['episode_reward'].rolling(rolling).mean()
            loaded = loaded.append(eval_loaded)
            loaded['date'] = date

            try:
                controller = config_yaml['env_override']['controller_config_file'].replace('jaco', '').replace('.json', '')
            except:
                controller = 'UNK'
            try:
                env_name = config_yaml['env_name']
            except:
                env_name = 'UNK'
            try:
                img_obs = config_yaml['env_override']['use_camera_obs']
                object_obs = config_yaml['env_override']['use_object_obs']
                proprio_obs = config_yaml['use_proprio_obs']
            except:
                img_obs = True
                object_obs = False
                proprio_obs = False
            kinematic_type = "None"
            try:
                kinematic_type = config_yaml['agent']['kinematic_type']
            except:
                try:
                    kinematic_type = config_yaml['agent']['experiment_type']
                except:
                    kk = 'use_kinematic_loss'
                    if kk in config_yaml['agent'].keys():
                        kinematic_type = config_yaml['agent']['use_kinematic_loss']
                        if kinematic_type == 1:
                            kinematic_type = 'loss'
                        if kinematic_type == 0:
                            kinematic_type = 'None'

            if img_obs:
                obs_type = 'IMG'
            if proprio_obs and object_obs:
                obs_type = 'STE'


            loaded['controller'] = controller
            loaded['kinematic_type'] = kinematic_type
            loaded['use_img_obs'] = int(img_obs)
            loaded['use_object_obs'] = int(object_obs)
            loaded['use_proprio_obs'] = int(proprio_obs)
            loaded['name'] = exp_id +  env_name +  obs_type + '_KINE' + kinematic_type + difficulty+controller + str(day)
            if start:
                start = False
                data = loaded
            else:
                data = data.append(loaded)
                print('adding', task, data.shape)

    if loaded.shape[0]:
        fig = px.line(data, x='step', y='episode_reward_smooth', color='name', symbol='date', width=2800, height=800)
        fig.update_layout(height=500, width=1400)
        fig.update_traces(mode="markers+lines")
        fig.write_html('results_'+task+'.html')

        img_data = data[data['use_img_obs'] == True]
        fig = px.line(img_data, x='step', y='episode_reward_smooth', color='name', symbol='date', width=2800, height=800)
        fig.update_layout(height=500, width=1400)
        fig.update_traces(mode="markers+lines")
        fig.write_html('results_'+task+'_img.html')

        state_data = data[data['use_img_obs'] == False]
        fig = px.line(img_data, x='step', y='episode_reward_smooth', color='name', symbol='date', width=2800, height=800)
        fig.update_layout(height=500, width=1400)
        fig.update_traces(mode="markers+lines")
        fig.write_html('results_'+task+'_state.html')





