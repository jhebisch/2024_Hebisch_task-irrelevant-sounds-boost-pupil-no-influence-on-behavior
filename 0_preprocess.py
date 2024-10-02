import os, glob, datetime
from functools import reduce
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm
from IPython import embed as shell

import utils
import utils_amsterdam_contrast_detection
import utils_amsterdam_orientation_discrimination

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()
sns.set_palette("tab10")

project_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(project_dir, 'data')
figs_dir = os.path.join(project_dir, 'figs')
n_jobs = 8

data_sets = [
            'amsterdam_contrast_detection',
            'amsterdam_orientation_discrimination',
            ]
for data_set in data_sets:

    edf_filenames = glob.glob(os.path.join(data_dir, data_set, '*.edf'))
    print(edf_filenames)
    if data_set == 'amsterdam_contrast_detection':
        res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(utils_amsterdam_contrast_detection.load_amsterdam_contrast_detection)(filename, figs_dir) 
                                                                 for filename in tqdm(edf_filenames))
    if data_set == 'amsterdam_orientation_discrimination':
        res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(utils_amsterdam_orientation_discrimination.load_amsterdam_orientation_discrimination)(filename, figs_dir) 
                                                                 for filename in tqdm(edf_filenames))

    # unpack:
    df = pd.concat([res[i][0] for i in range(len(res))]).reset_index(drop=True)
    epochs_p_stim = pd.concat([res[i][1] for i in range(len(res))])
    epochs_p_resp = pd.concat([res[i][2] for i in range(len(res))])

    # get all variables in right:

    if data_set == 'amsterdam_contrast_detection':

        df['block_type'] = df['block_type'].map({'r': 0, 'f': 1}).astype(int)
        df['condition'] = (df['condition']=='boost').astype(int)
        df['stimulus'] = df['stimulus'].map({'absent': 0, 'present': 1}).astype(int)
        df['choice'] = 0
        df.loc[(df['stimulus']==1)&(df['correct']==1), 'choice'] = 1
        df.loc[(df['stimulus']==0)&(df['correct']==0), 'choice'] = 1
        df['block_id'] = df['block_id'].astype(int)-1
        df['date'] = pd.to_datetime(df['start_time']).dt.date
        df['block_split'] = (df['trial_nr']>=185).map({False: 1, True: 2}) # FIXME 
        df.loc[df['trial_nr']<=9, 'block_split'] = 0
        
    if data_set == 'amsterdam_orientation_discrimination':

        # rename:
        df = df.rename({'DV_   {}'.format(i): 'dv{}'.format(i) for i in range(8)}, axis=1)

        # # zscore dvs:
        # df = df.groupby(['subject_id', 'block_id']).apply(zscore_dvs)

        # variables:
        df['condition'] = df['condition'].map({'normal': 0, 'AS_0': 1, 'AS_1': 2, 'AS_2': 3}).astype(int)
        df['dv'] = df.loc[:,['dv{}'.format(i) for i in range(8)]].mean(axis=1)
        df['dv_abs'] = df['dv'].abs()
        df['stimulus'] = np.NaN
        df.loc[df['dv']>0, 'stimulus'] = 1
        df.loc[df['dv']<0, 'stimulus'] = 0
        df.loc[(df['stimulus']==0)&(df['correct']==1), 'choice'] = 0
        df.loc[(df['stimulus']==0)&(df['correct']==0), 'choice'] = 1
        df.loc[(df['stimulus']==1)&(df['correct']==1), 'choice'] = 1
        df.loc[(df['stimulus']==1)&(df['correct']==0), 'choice'] = 0

        # add pupil:
        x = np.array(epochs_p_stim.columns, dtype=float)
        df['pupil_b'] = epochs_p_stim.loc[:,(x>-2)&(x<-1)].mean(axis=1).values
        df['pupil_r'] = epochs_p_resp.loc[:,(x>1.5)&(x<2.5)].mean(axis=1).values - df['pupil_b']

    # add number of trials:
    df['nr_trials_total'] = df.groupby(['subject_id'])['trial_nr'].transform('count')
    print(df.groupby(['subject_id'])['nr_trials_total'].mean())

    # add accuracy per block:
    def compute_accuracy(df, trial_nr=9):
        df['correct_avg'] = df.loc[(df['trial_nr']>trial_nr)&(df['condition']==0)&(df['correct']!=-1), 'correct'].mean()
        return df
    df = df.groupby(['subject_id', 'block_id'], group_keys=False).apply(compute_accuracy, 9)
    print(df.groupby(['subject_id', 'block_id'])['correct_avg'].mean())

    # save:
    print('saving... {}'.format(data_set))
    df.to_csv(os.path.join(data_dir, '{}_df.csv'.format(data_set)))
    epochs_p_stim.to_hdf(os.path.join(data_dir, '{}_epochs_p_stim.hdf'.format(data_set)), key='pupil')
    epochs_p_resp.to_hdf(os.path.join(data_dir, '{}_epochs_p_resp.hdf'.format(data_set)), key='pupil')