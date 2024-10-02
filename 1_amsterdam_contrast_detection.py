'''
Second and final script to be applied for analysis of Experiment 1.
Handles exclusions.
Adds quantification metrics of pupil size to behavioral dataframe.
Plots and gives statistics for the manuscript.

Input:
CSV files of behavioral data and eye data epochs that were saved after preprocessing.

Output:
Plots to figure folder.

'''

# imports
import os, glob, datetime
from functools import reduce
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm
from IPython import embed as shell
from pingouin import bayesfactor_ttest
from pingouin import ttest
import math
import statistics
import statsmodels.api as sm
import statsmodels.formula.api as smf

import utils
import utils_amsterdam_contrast_detection

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
    # 'lines.linewidth': 0.8,
    'ytick.color':'Black',})
sns.plotting_context()
color_noAS = '#DECA59' #'#CBC034' 
color_AS = '#6CA6F0'
colorpalette_noAS = "blend:#DECA59,#DE6437"
color_blend = "blend:#6CA6F0,#702D96"
sns.set_palette(color_blend, n_colors=3)

# define folders
project_dir = 'G:\\.shortcut-targets-by-id\\1-KxMllxLjzCUetyGI01Gp0sVPupwLl38\\2023_as_multiple_datasets' # os.path.dirname(os.getcwd()) #TODO change back in the end
data_dir = os.path.join(project_dir, 'data')
figs_dir = os.path.join(project_dir, 'figs') 
data_set = 'amsterdam_contrast_detection'
trial_cutoff = 9

# load data:
print('loading... {}'.format(data_set))
df = pd.read_csv(os.path.join(data_dir, '{}_df.csv'.format(data_set)))
epochs_p_stim = pd.read_hdf(os.path.join(data_dir, '{}_epochs_p_stim.hdf'.format(data_set)), key='pupil')
epochs_p_resp = pd.read_hdf(os.path.join(data_dir, '{}_epochs_p_resp.hdf'.format(data_set)), key='pupil')

# exclude trials:
exclude = np.array( 
                    (df['correct'] == -1) |
                    (df['rt']<0.2) |
                    (df['rt']>4.5) |
                    (df['blinks_dec']>.2)
                    )
df = df.loc[~exclude,:].reset_index(drop=True)
epochs_p_stim = epochs_p_stim.loc[~exclude,:]
epochs_p_resp = epochs_p_resp.loc[~exclude,:]

# exclude subjects if their trial number falls under 1000:
print('# subjects before exclusion = {}'.format(len(df['subject_id'].unique())))
counts = df.groupby(['subject_id'])['trial_nr'].count().reset_index()
ind = df['subject_id'].isin(counts.loc[counts['trial_nr']>1000,'subject_id']).values
df = df.loc[ind,:].reset_index(drop=True)
epochs_p_stim = epochs_p_stim.loc[ind,:]
epochs_p_resp = epochs_p_resp.loc[ind,:]
print('# subjects after exclusion = {}'.format(len(df['subject_id'].unique())))

# add sequential info:
df = df.groupby(['subject_id', 'block_id']).apply(utils_amsterdam_contrast_detection.compute_sequential).reset_index(drop=True)
df.loc[df['stimulus_p']==0, 'stimulus_p'] = -1
df.loc[df['choice_p']==0, 'choice_p'] = -1

# add pupil:
## pre-trial baseline (bins)
x = np.array(epochs_p_stim.columns, dtype=float)
df['pupil_b'] = epochs_p_stim.loc[:,(x>-1.5)&(x<-1)].mean(axis=1).values
bins = 3
df['pupil_b_bin'] = df.groupby(['subject_id', 'block_id'])['pupil_b'].apply(pd.qcut, q=bins, labels=False)

## baseline used for task-evoked pupil response quantification
x = np.array(epochs_p_stim.columns, dtype=float)
df['pupil_tpr_b'] = epochs_p_stim.loc[:,(x>-0.5)&(x<0)].mean(axis=1).values

## task-evoked pupil response (TPR)
x = np.array(epochs_p_resp.columns, dtype=float)
df['tpr'] = epochs_p_resp.loc[:,(x>-0.5)&(x<1.5)].mean(axis=1).values - df['pupil_tpr_b'] # use this baseline instead of pupil_b?
### correct phasic pupil responses for reaction time effects:
df = df.groupby(['subject_id', 'block_id', 'condition'], group_keys=False).apply(utils.bin_correct_tpr, trial_cutoff=trial_cutoff)
### create 8 TPR bins:
bins = 8
df['tpr_bin'] = df.groupby(['subject_id', 'block_id', 'condition'])['tpr_c'].apply(pd.qcut, q=bins, labels=False)
# df['tpr_bin'] = df.groupby(['subject_id', 'block_id', 'condition'],sort=False)['tpr_c'].apply(pd.qcut, q=bins, labels=False).reset_index().set_index('level_3').drop(['subject_id', 'block_id', 'condition'] ,axis=1)

## pupil response to task-irrelevant stimuli
res = epochs_p_stim.groupby(['subject_id', 'block_id'], sort=False).apply(utils_amsterdam_contrast_detection.compute_pupil_response_diff)
df['pupil_r_diff'] = res['pupil_r_diff'].values
bins = 8
df['pupil_r_diff_bin'] = -1 # np.NaN
df.loc[df['condition']==1,'pupil_r_diff_bin'] = df.loc[df['condition']==1].groupby(['subject_id', 'block_id'])['pupil_r_diff'].apply(pd.qcut, q=bins, labels=False)

## compute mean pupil value on normal trials over time window used for task-irrelevant stimulus-evoked pupil response
res = epochs_p_stim.groupby(['subject_id', 'block_id'], sort=False).apply(utils_amsterdam_contrast_detection.compute_pupil_scalars, which='base')
df['pupil_base'] = res['pupil_base'].values

## compute mean pupil value on task-irrelevant stimulus trials over time window used for task-irrelevant stimulus-evoked pupil response
res = epochs_p_stim.groupby(['subject_id', 'block_id'], sort=False).apply(utils_amsterdam_contrast_detection.compute_pupil_scalars, which='r_stim')
df['pupil_r_stim'] = res['pupil_r_stim'].values

# # baseline:
# epochs_p_stim = epochs_p_stim - np.atleast_2d(df['pupil_b'].values).T
# epochs_p_resp = epochs_p_resp - np.atleast_2d(df['pupil_b'].values).T


################################ Figures and analyses ############################

# -----------------------------------------------------------------------
# Fig.2: Pupil responses (endogenous and exogenous)
# -----------------------------------------------------------------------
## variability in task-evoked pupil response (tpr) (fig. 2A)
### baseline
epochs_p_resp = epochs_p_resp - np.atleast_2d(df['pupil_tpr_b'].values).T
### add task-evoked pupil response bin to index of dataframe for epochs locked to response (button-press)
epochs_p_resp = epochs_p_resp.reset_index()
epochs_p_resp['tpr_bin'] = df['tpr_bin']
columns = ['subject_id', 'block_id', 'trial_nr', 'condition', 'tpr_bin']
epochs_p_resp = epochs_p_resp.set_index(columns)

x = epochs_p_resp.columns
fig = utils_amsterdam_contrast_detection.plot_normal_pupil(epochs_p_resp.query('trial_nr > {}'.format(trial_cutoff)).loc[:, (x>-3)&(x<2)], locking='resp', draw_box=True, shade=False)
# fig.savefig(os.path.join(figs_dir, '{}_normal_pupil_resp.pdf'.format(data_set)))
## mean and sem tpr in highest and lowest bin
print(df.loc[(df['tpr_bin']==7) & (df['condition']==0), 'tpr_c'].mean())
print(df.loc[(df['tpr_bin']==0) & (df['condition']==0), 'tpr_c'].mean())
print(df.loc[(df['tpr_bin']==7) & (df['condition']==0), 'tpr_c'].sem())
print(df.loc[(df['tpr_bin']==0) & (df['condition']==0), 'tpr_c'].sem())

## pupil response to task-irrelevant stimulus (fig. 2B)
### curve
x = epochs_p_stim.columns
fig = utils_amsterdam_contrast_detection.plot_pupil_responses(epochs_p_stim.query('trial_nr > {}'.format(trial_cutoff)).loc[:, (x>-1)&(x<4)])
# fig.savefig(os.path.join(figs_dir, '{}_pupil_responses.pdf'.format(data_set)))

### scatterplot of fig. 2C -> see code for figure 3

# -----------------------------------------------------------------------
# Fig.3 (+S4 and S5): Bias effects (endogenous and in response to task-irrelevant stimulus) + behavioural effects of task-irrelevant stimulus
# -----------------------------------------------------------------------
## correlation task-evoked pupil-linked arousal and behaviour
fig, reg_result, t_value_tpr = utils.plot_tpr_behav_cor(df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'block_type', 'tpr_bin'], indices=['c_abs'], r_place=5)
reg_result.summary()
# fig.savefig(os.path.join(figs_dir, '{}_tpr_c_abs.pdf'.format(data_set)))

## TPR correlation split for rare and frequent block type (fig. S4)
df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'block_type', 'tpr_bin'])
df_res = df_res.loc[df_res['condition']==0,:]

for m in ['d', 'c', 'c_abs','rt']:
    fig = utils_amsterdam_contrast_detection.plot_split_tpr_behav_cor(df_res, m=m)
    # fig.savefig(os.path.join(figs_dir, '{}_tpr_behaviour_{}_split.pdf'.format(data_set, m)), bbox_inches='tight')

## compare pupil resps (task-irrelevant stimulus and task evoked) and behaviour
groupby=['subject_id', 'condition', 'block_type']
df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=groupby)

for m in ['c_abs']:
    fig = utils_amsterdam_contrast_detection.plot_compare(df_res, y=m, comp_x1='pupil_base', comp_x2='pupil_r_stim')
    # fig.savefig(os.path.join(figs_dir, '{}_compare_behavior_{}_p_resps.pdf'.format(data_set, m)))

## task-irrelevant stimulus-evoked effects in behaviour
df_res = utils.compute_results(df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'block_type'])

for m in ['rt', 'correct', 'd', 'c', 'c_abs','pupil_r_diff']:
    fig = utils_amsterdam_contrast_detection.plot_se_behav(df_res, m, data_set=data_set) # split by block_type (rare vs. frequent vis stim)
    # fig.savefig(os.path.join(figs_dir, '{}_as_behavior_{}_block_type.pdf'.format(data_set, m)))
    fig = utils_amsterdam_contrast_detection.plot_se_behav(df_res, m, data_set=data_set, split=False)
    # fig.savefig(os.path.join(figs_dir, '{}_as_behavior_{}.pdf'.format(data_set, m)))

# ------------------------------------------------------------------------
# Fig. 5: baseline pupil size and task-irrelevant stimulus effect interaction 
# ------------------------------------------------------------------------
df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'block_type', 'pupil_b_bin'])
df_res1=df_res.loc[df_res['condition']==1].reset_index()
df_res2=df_res.loc[df_res['condition']==0].reset_index()
df_res1['c_abs2']=df_res2['c_abs']
df_res1['c_abs_diff']=df_res1['c_abs']-df_res1['c_abs2']

for m,n,y,z in zip(['c_abs_diff', 'pupil_r_diff'], [df_res1, df_res],[-0.07,-0.6], [[-0.08,0.1],[-1,5.5]]):
    fig = utils_amsterdam_contrast_detection.baseline_interaction(m,n,y,z)
    # fig.savefig(os.path.join(figs_dir, '{}_{}_on_baseline_bin.pdf'.format(data_set, m)))

# ------------------------------------------------------------------------
# Fig S1: means of behaviour
# ------------------------------------------------------------------------
df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id'])
df_res1 = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'block_type'])

for m in ['c_abs','pupil_r_diff','d','c','rt']:
    fig = utils_amsterdam_contrast_detection.plot_means(df_res, measure=m, data_set=data_set)
    # fig.savefig(os.path.join(figs_dir, '{}_{}_mean.pdf'.format(data_set, m)))

for m in ['c_abs','pupil_r_diff','d','c','rt']:
    fig = utils_amsterdam_contrast_detection.plot_means(df_res1, measure=m, data_set=data_set, split=True)
    # fig.savefig(os.path.join(figs_dir, '{}_{}_mean_split.pdf'.format(data_set, m)))

# ----------------------------------------------------------------------
# Fig S2: pupil size metrics (baseline, TPR, task-irrelevant sound evoked pupil response) across trials (normal trials)
# -----------------------------------------------------------------------
## tpr and bl across trials by condition:
n_jobs = 4
n_boot = 100
means = []
sems = []
for c in [0,1]:
    means.append(utils.compute_results(df.loc[df['condition']==c,:], groupby=['trial_nr']))
    sem_ = pd.concat(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(utils.compute_results)
                                                (df.loc[df['condition']==c,:].sample(frac=1, replace=True), ['trial_nr'], i)
                                                for i in range(n_boot)))
    sems.append((sem_.groupby(['trial_nr']).quantile(0.84) - sem_.groupby(['trial_nr']).quantile(0.16))/2)

for y in ['rt', 'correct', 'd', 'c', 'c_abs', 'pupil_b', 'tpr', 'pupil_r_diff']:
    fig = utils_amsterdam_contrast_detection.plot_across_trials(y, means, sems, trial_cutoff)
    # fig.savefig(os.path.join(figs_dir, '{}_across_trials_{}.pdf'.format(data_set, y)))

# -----------------------------------------------------------------------
# Fig. S3 task-evoked pupil response and behaviour
# -----------------------------------------------------------------------
fig2, reg_result2, t_value_tpr2 = utils.plot_tpr_behav_cor(df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'block_type', 'tpr_bin'], indices=['c'], r_place=5)
# fig2.savefig(os.path.join(figs_dir, '{}_tpr_c.pdf'.format(data_set)))
fig3, reg_result3, t_value_tpr3 = utils.plot_tpr_behav_cor(df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'block_type', 'tpr_bin'], indices=['d'], r_place=5)
# fig3.savefig(os.path.join(figs_dir, '{}_tpr_d.pdf'.format(data_set)))
fig4, reg_result4, t_value_tpr4 = utils.plot_tpr_behav_cor(df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'block_type', 'tpr_bin'], indices=['rt'], r_place=5)
# fig4.savefig(os.path.join(figs_dir, '{}_tpr_rt.pdf'.format(data_set)))

# ------------------------------------------------------------------------
# Fig. S6 Variability of task-irrelevant stimulus-evoked pupil responses
# ------------------------------------------------------------------------
# correlation size of task-irrelevant evoked pupil response (by bin) and behaviour 
groupby=['subject_id', 'condition', 'block_type', 'pupil_r_diff_bin']
df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=groupby)
df_res['c_abs_diff'] = np.NaN
for s in df_res['subject_id'].unique():
    for b in df_res['block_type'].unique():
        df_res.loc[(df_res['subject_id']==s) & (df_res['block_type']==b),'c_abs_diff'] = df_res.loc[(df_res['subject_id']==s) & (df_res['block_type']==b),'c_abs'] - float(df_res.loc[ (df_res['subject_id']==s) & (df_res['block_type']==b) & (df_res['condition']==0), 'c_abs'].values)
df_res = df_res.loc[df_res['condition']==1,:]

fig= utils_amsterdam_contrast_detection.pupil_tis_resp_behav_cor(df_res=df_res, indices=['c_abs_diff'], r_place=5)
# fig.savefig(os.path.join(figs_dir, '{}_as_size_behavior_{}.pdf'.format(data_set, m)))
