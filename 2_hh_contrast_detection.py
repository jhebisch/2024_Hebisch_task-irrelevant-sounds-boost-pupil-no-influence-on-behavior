'''
Second and final script to be applied for analysis of Experiment 2.
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
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm
import statsmodels.api as sm
import statsmodels.formula.api as smf

import utils
import utils_hh_contrast_detection

# Style settings for plots
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
    'ytick.color':'Black',} )
sns.plotting_context()
color_noAS = '#DECA59' #'#CBC034' 
color_AS = '#6CA6F0'
colorpalette_noAS = "blend:#DECA59,#DE6437"
color_blend = "blend:#6CA6F0,#702D96"
sns.set_palette(color_blend, n_colors=9)

# define folders
project_dir = 'C:\\Users\\Josefine\\Documents\\Promotion\\pilot_study\\arousal_percept_experiments' # os.path.dirname(os.getcwd())
data_dir = os.path.join(project_dir, 'data', 'contrast_detection') # os.path.join(project_dir, 'data')
figs_dir = os.path.join(data_dir, 'figs', 'group_figs') # os.path.join(project_dir, 'figs')
data_set = 'hh_contrast_detection'
trial_cutoff = 11 # refers to trial number

# load data:
print('loading... {}'.format(data_set))
df = pd.read_csv(os.path.join(data_dir, '{}_df_meta.csv'.format(data_set)))
epochs_p_sound = pd.read_hdf(os.path.join(data_dir, '{}_epochs_p_sound.hdf'.format(data_set)), key='pupil')
epochs_p_resp = pd.read_hdf(os.path.join(data_dir, '{}_epochs_p_resp.hdf'.format(data_set)), key='pupil')
epochs_p_dphase = pd.read_hdf(os.path.join(data_dir, '{}_epochs_p_dphase.hdf'.format(data_set)), key='pupil')

# exclude trials:
exclude = np.array( 
                    (df['correct'] == -1) |
                    (df['rt']<0.2) |
                    (df['rt']>4.5) | 
                    (df['blinks_dec']>.2)
                    )
df = df.loc[~exclude,:].reset_index(drop=True)
epochs_p_sound = epochs_p_sound.loc[~exclude,:]
epochs_p_resp = epochs_p_resp.loc[~exclude,:]
epochs_p_dphase = epochs_p_dphase.loc[~exclude,:]

# exclude subjects if their trial number falls under 840 (70%):
print('# subjects before exclusion = {}'.format(len(df['subject_id'].unique())))
counts = df.groupby(['subject_id'])['trial_nr'].count().reset_index()
ind = df['subject_id'].isin(counts.loc[counts['trial_nr']>840,'subject_id']).values
df = df.loc[ind,:].reset_index(drop=True)
epochs_p_sound = epochs_p_sound.loc[ind,:]
epochs_p_resp = epochs_p_resp.loc[ind,:]
epochs_p_dphase = epochs_p_dphase.loc[ind,:]
print('# subjects after exclusion = {}'.format(len(df['subject_id'].unique())))

# add sequential info:
df = df.groupby(['subject_id', 'block_id']).apply(utils_hh_contrast_detection.compute_sequential).reset_index(drop=True)
df.loc[df['stimulus_p']==0, 'stimulus_p'] = -1
df.loc[df['choice_p']==0, 'choice_p'] = -1

# add pupil:
## pre-trial pupil baseline
z = np.array(epochs_p_dphase.columns, dtype=float)
df['pupil_b'] = np.atleast_2d(epochs_p_dphase.loc[:,(z>-3.5)&(z<-3)].mean(axis=1)).T #time frame is half a second before anything (including potential white noise) happens
bins = 3
df['pupil_b_bin'] = df.groupby(['subject_id', 'block_id'])['pupil_b'].apply(pd.qcut, q=bins, labels=False)

## baseline used for task-evoked pupil response quantification
z = np.array(epochs_p_dphase.columns, dtype=float)
df['pupil_tpr_b'] = np.atleast_2d(epochs_p_dphase.loc[:,(z>-0.5)&(z<0)].mean(axis=1)).T #time frame used for baseline??

## task-evoked pupil response
x = np.array(epochs_p_resp.columns, dtype=float)
df['tpr'] = epochs_p_resp.loc[:,(x>-0.5)&(x<1.5)].mean(axis=1).values - df['pupil_tpr_b'] #this baseline?
### correct phasic pupil responses for reaction time effects:
df = df.groupby(['subject_id', 'block_id', 'condition'], group_keys=False).apply(utils.bin_correct_tpr, trial_cutoff=trial_cutoff)
### create 8 TPR bins:
bins = 8
df['tpr_bin'] = df.groupby(['subject_id', 'block_id', 'condition'])['tpr_c'].apply(pd.qcut, q=bins, labels=False)

# baselines (again) to apply to epochs dataframes directly: 
y = np.array(epochs_p_sound.columns, dtype=float)
z = np.array(epochs_p_dphase.columns, dtype=float)

stim_bl = np.atleast_2d(epochs_p_sound.loc[:,(y>-0.5)&(y<0)].mean(axis=1)).T
pupil_b = np.atleast_2d(epochs_p_dphase.loc[:,(z>-3.5)&(z<-3)].mean(axis=1)).T 
pupil_tpr_b = np.atleast_2d(epochs_p_dphase.loc[:,(z>-0.5)&(z<0)].mean(axis=1)).T 

epochs_p_resp = epochs_p_resp - pupil_tpr_b 
epochs_p_sound = epochs_p_sound - stim_bl
epochs_p_dphase = epochs_p_dphase - pupil_b

# create epochs locked to decision phase with substracted mean of normal condition to make task-irrelevant sound influence visible
epochs_diff_sound, epoch_diff_dphase = utils_hh_contrast_detection.create_diff_epochs(epoch_dphase_in= epochs_p_dphase.query('trial_nr > {}'.format(trial_cutoff)), groupby=['subject_id', 'block_id']) #here, I first used mean normal trials by participant, code now is by participant by block
# add baseline to epoch diff stim
y = np.array(epochs_diff_sound.columns, dtype=float)
diff_sound_bl = np.atleast_2d(epochs_diff_sound.loc[:,(y>-0.5)&(y<0)].mean(axis=1)).T
epochs_diff_sound = epochs_diff_sound - diff_sound_bl

## pupil response to task-irrelevant stimuli
res = utils_hh_contrast_detection.compute_pupil_response_diff(epochs_diff_sound, epochs_p_sound)
df['pupil_r_diff'] = res['pupil_r_diff'].values
bins = 8
df['pupil_r_diff_bin'] = -1 
df.loc[df['condition']==1,'pupil_r_diff_bin'] = df.loc[df['condition']==1].groupby(['subject_id', 'block_id'], group_keys=False)['pupil_r_diff'].apply(pd.qcut, q=bins, labels=False)

### create fake SOAs for normal condition (relevant for pupil response calculation for figure 3B)
df['fake_soa'] = np.NaN
df.loc[df['condition']==0,'fake_soa'] = np.random.uniform(-3,0.5,len(df.loc[df['condition']==0]))

res = utils_hh_contrast_detection.compute_pupil_scalars(epochs=epochs_p_dphase, df=df, condition="normal")
df['pupil_base'] = res.values
df.loc[(df['condition']==1),'pupil_base']=np.NaN
res = utils_hh_contrast_detection.compute_pupil_scalars(epochs=epochs_p_dphase, df=df, condition="boost")
df['pupil_r_sound'] = res.values
df.loc[(df['condition']==0),'pupil_r_sound']=np.NaN

#################### Figures and analyses ###############################

# -----------------------------------------------------------------------
# Fig.2: Pupil responses (endogenous and exogenous)
# -----------------------------------------------------------------------
## variability in task-evoked pupil response (tpr) (fig. 2A)
### add task-evoked pupil response bin to index of dataframe for epochs locked to response (button-press)
epochs_p_resp = epochs_p_resp.reset_index()
epochs_p_resp['tpr_bin'] = df['tpr_bin']
columns = ['subject_id', 'block_id', 'trial_nr', 'condition', 'actual_soa', 'soa_bin','tpr_bin']
epochs_p_resp = epochs_p_resp.set_index(columns)

x = epochs_p_resp.columns
fig = utils_hh_contrast_detection.plot_normal_pupil(epochs_p_resp.iloc[:,(x>-3)&(x<2)].query('trial_nr > {}'.format(trial_cutoff)), locking='resp', draw_box=True, shade=False)
# fig.savefig(os.path.join(figs_dir, '{}_normal_pupil_resp.pdf'.format(data_set)))
## mean and sem tpr in highest and lowest bin
print(df.loc[(df['tpr_bin']==7) & (df['condition']==0), 'tpr_c'].mean())
print(df.loc[(df['tpr_bin']==0) & (df['condition']==0), 'tpr_c'].mean())
print(df.loc[(df['tpr_bin']==7) & (df['condition']==0), 'tpr_c'].sem())
print(df.loc[(df['tpr_bin']==0) & (df['condition']==0), 'tpr_c'].sem())

## pupil response task-irrelevant stimulus (fig. 2B)
### curves with sliding window for SOA of task-irrelevant sound onset
fig = utils_hh_contrast_detection.plot_pupil_sliding_bin(epochs_p_dphase.query('trial_nr > {}'.format(trial_cutoff)), epoch_diff_dphase, windowsize=0.8, stepsize=0.35, ylim=(-1, 3.2))
# fig.savefig(os.path.join(figs_dir, '{}_pupil_responses.pdf'.format(data_set)))
# fig.savefig(os.path.join(figs_dir, '{}_pupil_responses_legend.pdf'.format(data_set)), bbox_inches='tight')

### task-irrelevant sound evoked pupil respons values per participant scatterplot (fig. 2C): 
fig, df_means, m, t_values = utils_hh_contrast_detection.plot_p_stats_sliding_bin(epochs_diff_sound, windowsize=0.8, stepsize=0.35, maxi=2, ylim=None)
# fig.savefig(os.path.join(figs_dir, '{}_pupil_responses_stats.pdf'.format(data_set)))
print(m.fit())
print(df_means.groupby(['variable']).value.mean())

# -----------------------------------------------------------------------
# Fig.3 (+S5): Bias effects (endogenous and in response to task-irrelevant stimulus) + other behavioural effects of task-irrelevant stimulus
# -----------------------------------------------------------------------
## correlation task-evoked pupil-linked arousal and behaviour
fig, reg_results, t_value_tpr = utils.plot_tpr_behav_cor(df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'tpr_bin'], indices=['c_abs']) #, method='PermutationMethod')
# fig.savefig(os.path.join(figs_dir, '{}_tpr_c_abs.pdf'.format(data_set)))

## compare pupil resps (task-irrelevant stimulus and task evoked) and behaviour
fig = utils_hh_contrast_detection.plot_compare(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition'], m = 'c_abs')
# fig.savefig(os.path.join(figs_dir, '{}_compare_behavior_{}_p_resps.pdf'.format(data_set, m)))

## task-irrelevant stimulus-evoked effects in behaviour (compute results with sliding window)
for m in ['rt', 'd', 'c', 'c_abs']: # 'correct',
    fig, df_diffs, _ = utils_hh_contrast_detection.plot_res_sliding_bin(df, var=m, windowsize=0.8, stepsize=0.35)
    # fig.savefig(os.path.join(figs_dir, '{}_as_behavior_{}.pdf'.format(data_set, m)))

# --------------------------------------------------------------------
# Fig. 5: baseline pupil size and task-irrelevant stimulus effect interaction
# --------------------------------------------------------------------
## coarser SOA bins
df['cond_SOA'] = np.NaN
bins=[-3.043, -2.125, -1.25, -.375, 0.51] # added a little buffer for sound that was played a little too early or a little too late #[-3, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6]
bin_names=[1,2,3,4]
df['cond_SOA']= pd.cut(df['actual_soa'], bins, labels=bin_names)
df['cond_SOA']=df['cond_SOA'].astype('float').fillna(0).astype('int') #make nans zeros for simplicity of depicting no stim categories in the end
# df.loc[(df['condition']=='boost') & (df['actual_soa'].isnull()),'cond_SOA']=np.NaN

## two way ANOVA pre-trial baselines
df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'cond_SOA', 'pupil_b_bin'])
df_res0=df_res.loc[df_res['cond_SOA']==0].reset_index()
df_res1=df_res.loc[df_res['cond_SOA']!=0].reset_index()
for s in df_res1['subject_id'].unique():
    for b in df_res1['pupil_b_bin'].unique():
        df_res1.loc[(df_res1['subject_id']==s) & (df_res1['pupil_b_bin']==b),'c_abs_diff'] = df_res1.loc[(df_res1['subject_id']==s) & (df_res1['pupil_b_bin']==b),'c_abs'] - df_res0.loc[(df_res0['subject_id']==s) & (df_res0['pupil_b_bin']==b),'c_abs'].values

for m,n,y,z in zip(['pupil_r_diff', 'c_abs_diff'], [df_res, df_res1],[-2.5,-0.12],[[-4.6,5.1],[-0.17,0.05]]): ## define m
    fig = utils_hh_contrast_detection.baseline_interaction(m=m, n=n, y=y, z=z)
    # fig.savefig(os.path.join(figs_dir, '{}_{}_on_baseline_bin.pdf'.format(data_set, m)))

# ------------------------------------------------------------------------
# Fig S1: means of behaviour
# ------------------------------------------------------------------------
df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id'])

for m in ['c_abs','pupil_r_diff','d','c','rt']:
    fig = utils_hh_contrast_detection.plot_means(df_res, measure=m)
    # fig.savefig(os.path.join(figs_dir, '{}_{}_mean.pdf'.format(data_set, m)))

# -----------------------------------------------------------------------
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

for y in ['tpr', 'pupil_tpr_b', 'pupil_b', 'pupil_r_diff']:
    fig = utils_hh_contrast_detection.plot_new_across_trials(means, sems, y=y, trial_cutoff=trial_cutoff)
    # fig.savefig(os.path.join(figs_dir, '{}_across_trials_condition_{}.pdf'.format(data_set, y)))

## add-on (not in any figure): means and sems of measures per trial number heatmaps
means, sems = utils_hh_contrast_detection.vars_across_trials(df, windowsize = 0.8, stepsize = 0.35, n_jobs = 4, n_boot = 100)
for y in ['rt', 'correct', 'd', 'c', 'c_abs', 'pupil_tpr_b', 'pupil_b', 'tpr']:
    fig = utils_hh_contrast_detection.plot_across_trials(means, sems, y=y, trial_cutoff=trial_cutoff)
    # fig.savefig(os.path.join(figs_dir, '{}_across_trials_{}.pdf'.format(data_set, y)))

# -----------------------------------------------------------------------
# Fig. S3 task-evoked pupil response and behaviour
# -----------------------------------------------------------------------

fig, reg_results, t_value_tpr = utils.plot_tpr_behav_cor(df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'tpr_bin'], indices=['c'])
# fig.savefig(os.path.join(figs_dir, '{}_tpr_c.pdf'.format(data_set)))

fig2, reg_results2, t_value_tpr2 = utils.plot_tpr_behav_cor(df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'tpr_bin'], indices=['d'])
# fig2.savefig(os.path.join(figs_dir, '{}_tpr_d.pdf'.format(data_set)))

fig3, reg_results3, t_value_tpr3 = utils.plot_tpr_behav_cor(df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'tpr_bin'], indices=['rt'])
# fig3.savefig(os.path.join(figs_dir, '{}_tpr_rt.pdf'.format(data_set)))

# -----------------------------------------------------------------------
# Fig. S6 Variability of task-irrelevant stimulus-evoked pupil responses
# -----------------------------------------------------------------------
# correlation size of task-irrelevant evoked pupil response (by bin) and behaviour 
groupby=['subject_id', 'condition', 'pupil_r_diff_bin']
df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=groupby)
df_res['c_abs_diff'] = np.NaN
for s in df_res['subject_id'].unique():
    df_res.loc[(df_res['subject_id']==s),'c_abs_diff'] = df_res.loc[(df_res['subject_id']==s),'c_abs'] - float(df_res.loc[ (df_res['subject_id']==s) & (df_res['condition']==0), 'c_abs'].values)
df_res = df_res.loc[df_res['condition']==1,:]

fig= utils_hh_contrast_detection.pupil_tis_resp_behav_cor(df_res=df_res, indices=['c_abs_diff'], r_place=5)
# fig.savefig(os.path.join(figs_dir, '{}_as_size_behavior_{}.pdf'.format(data_set, m)))
