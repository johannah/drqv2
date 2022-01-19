import matplotlib
matplotlib.use("Agg")
import os
import sys
import plotly.io as pio
import plotly.express as px
import pandas as pd
from glob import glob
from IPython import embed
rolling = 10

for task in ['reach', 'lift', 'can', 'door']:
    train_paths = glob(os.path.join('exp_local', '*', '*%s*'%task, 'train.csv'))
    train_paths += glob(os.path.join('*_exp_local', 'exp_local', '*', '*%s*'%task, 'train.csv'))
    train_paths = sorted(train_paths)
    start = True
    for pp in train_paths:
        loaded = pd.read_csv(pp)
        loaded['phase'] = 'train'
        if loaded['step'].max() > 50000:
            exp_name = os.path.split(os.path.split(pp)[0])[1]
            date = os.path.split(os.path.split(os.path.split(pp)[0])[0])[1]
            day = int(date[-2:])
            if day >= 14:
                eval_loaded = pd.read_csv(pp)
                eval_loaded['phase'] = 'eval'
                eval_loaded['episode_reward_smooth']  = eval_loaded['episode_reward'].rolling(rolling).mean()
                base_name = date + exp_name
                loaded['episode_reward_smooth']  = loaded['episode_reward'].rolling(rolling).mean()
                loaded = loaded.append(eval_loaded)
                loaded['name'] = base_name
                loaded['date'] = date
                if start:
                    start = False
                    data = loaded
                else:
                    data = data.append(loaded)
                    print('adding', task, data.shape)

    if not start:
        fig = px.line(data, x='step', y='episode_reward_smooth', color='name', markers=True, symbol='phase', width=2800, height=800)
        pio.write_image(fig, 'results_'+task+'.png')



