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

####################### Currently not used #############################################

# # Fig S: plot AS consequences for rep probability and baseline
# for m in ['pupil_b', 'repetition_past', 'repetition_future', 'tpr','tpr_c']:
#     fig = utils_hh_contrast_detection.plot_history(df=df.loc[df['trial_nr']>trial_cutoff,:], m=m)
#     fig.savefig(os.path.join(figs_dir, '{}_history_bias_{}.pdf'.format(data_set, m)))

# # Fig. S3: Across subjects correlations(Overall c versus delta-c, Delta-pupil versus delta-c, Delta-pupil versus c)
# ## c versus delta-c #blue shadowing means cluster corrected significance, yellow means fdr adjusted significance, red means significance didn't hold up under any correction
# fig_ex10, df_cors_delta_c_ex10, df_corrrands_ex10, rand_max_clusters_ex10, p_values_cluster_ex10, true_clusters_ex10, fdr_adj_p = utils_hh_contrast_detection.sliding_window_cor_dev(df.loc[(df['trial_nr']>trial_cutoff)], measure='c', windowsize=0.8, stepsize=0.35, tail=-1)
# # fig_ex10.savefig(os.path.join(figs_dir, '{}_sliding_cor_c_deltac.pdf'.format(data_set)))

# ## example plot for correlation (delta boost-normal)
# df_sdt_cond = df.groupby(['subject_id', 'condition']).apply(utils_hh_contrast_detection.sdt)
# df_sdt_cond.reset_index(inplace=True)
# df_sdt_cond = df_sdt_cond.rename(columns = {'index':'condition'})

# fig = plt.figure(figsize=(2,2))
# ax = fig.add_subplot() #121)
# y = 'c'
# x = 'c'
# y1 = df_sdt_cond.loc[(df_sdt_cond['condition']==0)].reset_index()
# y2 = df_sdt_cond.loc[(df_sdt_cond['condition']==1)].reset_index()

# y2['diff'] = y2[y] - y1[y]
# y2['diff_abs'] = abs(y2['diff'].copy())

# plt.axhline(0, color='k', lw=0.5, ls='dashed')

# sns.scatterplot(data=y2, x=x, y='diff') #, hue=z)
# sns.regplot(data=y2, x=x, y='diff', ax=ax, ci= None, scatter=False, color='k')
# truecorr, corrrand, p_value = utils_hh_contrast_detection.permutationTest_correlation(df_sdt_cond.loc[(df_sdt_cond['condition']==0), y].values, df_sdt_cond.loc[(df_sdt_cond['condition']==1), y].values, tail=-1) #TODO fix

# ax.text(0, -0.3, 'r = {}'.format(round(truecorr,2)), fontsize=9) # 'r = {}, p = {}'.format(round(truecorr,2),round(p_value,2))

# if (y == 'c'):
#     ax.set_ylabel('Δ Criterion c (AS - no AS)', fontsize=9)

# ax.set_xlabel('Criterion no-AS trials', fontsize=9)
# sns.despine(trim=True)
# plt.tight_layout()
# fig.savefig(os.path.join(figs_dir, '{}_cond_cor_c_deltac.pdf'.format(data_set)))

# ## sliding plot without corrections
# df_cors_delta_c = df_cors_delta_c_ex10.copy()
# measure = 'c'
# stepsize = 0.35
# fig, ax= plt.subplots(1,1,figsize=(3,2)) # figsize=(5,3.5))
# ax.axvline(0, color='k', ls='dashed', lw=0.5)
# # plt.plot(centers, true_corrs)
# sns.lineplot(data=df_cors_delta_c,x='centers', y='true_corrs', palette=sns.color_palette("blend:#6CA6F0,#7931A3", as_cmap=True))
# ax.set_xlabel('Center of AS SOA window (s)')
# if measure == 'c':
#     ax.set_ylabel('Correlation r') #(AS SOA window - no-AS condition)
#     ax.set_title('Sliding Window Correlation Criterion & Δ Criterion', fontweight='bold')
# else:
#     ax.set_ylabel('cor of {} & Δ {}'.format(measure, measure)) #(AS SOA window - no-AS condition)
# # shade area that is significant red, if it's still significant after cluster correction make it yellow
# start_span = list(df_cors_delta_c.iloc[np.where(df_cors_delta_c['significance'].diff()==1)]['centers'])
# end_span = list(df_cors_delta_c.iloc[np.where(df_cors_delta_c['significance'].diff()==-1)]['centers'])
# if (len(start_span)!=0) or (len(end_span)!=0):
#     if end_span[0] < start_span[0]:
#         start_span.insert(0,df_cors_delta_c['centers'][0])
#     for i,j in zip(start_span, end_span):
#         ax.axvspan(i,(j-stepsize), alpha=0.3)


# sns.despine(trim=True)
# plt.tight_layout()
# fig.savefig(os.path.join(figs_dir, '{}_sliding_cor_c_deltac_nocorrection.pdf'.format(data_set)))


# ## delta pupil vs c
# fig = utils_hh_contrast_detection.plot_sliding_cors(df, epochs_diff_sound, measure='c', windowsize=0.8, stepsize=0.35, maxi=2, sdt_groupby=['subject_id', 'in_bin'], delta=False)
# fig.savefig(os.path.join(figs_dir, '{}_sliding_cor_c_deltap.pdf'.format(data_set)))
# ## delta pupil vs delta c 
# fig = utils_hh_contrast_detection.plot_sliding_cors(df, epochs_diff_sound, measure='c', windowsize=0.8, stepsize=0.35, maxi=2, sdt_groupby=['subject_id', 'in_bin'], delta=True)
# fig.savefig(os.path.join(figs_dir, '{}_sliding_cor_deltac_deltap.pdf'.format(data_set)))

# # plot baseline pupil across trials:
# for y in ['rt', 'correct', 'pupil_tpr_b']:
#     mean = df.groupby(['subject_id', 'trial_nr']).mean().groupby(['trial_nr']).mean()[y]
#     sem = df.groupby(['subject_id', 'trial_nr']).mean().groupby(['trial_nr']).sem()[y]
#     fig = plt.figure(figsize=(2,2))
#     x = np.array(mean.index, dtype=float)
#     # plt.axvspan(9,52, color='blue', alpha=0.1)
#     # plt.axvspan(52,95, color='red', alpha=0.1)
#     plt.axvline(trial_cutoff, color='k', lw=0.5)
#     # plt.axvline(52, color='k', lw=0.5)
#     plt.fill_between(x, mean-sem, mean+sem, alpha=0.2)
#     plt.plot(x, mean)
#     plt.xlabel('Trial (#)')
#     plt.ylabel(y)
#     sns.despine(trim=True)
#     plt.tight_layout()
#     fig.savefig(os.path.join(figs_dir, '{}_across_trials_{}.pdf'.format(data_set, y)))

# # ---------------------
# # baseline interaction additional plots
# #--------------------------------------

# ## load in long baselines
# print('loading... {}'.format(data_set))
# # epochs_cp_stim = pd.read_hdf(os.path.join(data_dir, '{}_epochs_cp_stim.hdf'.format(data_set)), key='pupil')
# # epochs_cp_resp = pd.read_hdf(os.path.join(data_dir, '{}_epochs_cp_resp.hdf'.format(data_set)), key='pupil')
# # epochs_cp_dphase = pd.read_hdf(os.path.join(data_dir, '{}_epochs_cp_dphase.hdf'.format(data_set)), key='pupil')
# # epochs_cp_bl10 = pd.read_hdf(os.path.join(data_dir, '{}_epochs_cp_bl10.hdf'.format(data_set)), key='pupil')
# epochs_p_bl10 = pd.read_hdf(os.path.join(data_dir, '{}_epochs_p_bl10.hdf'.format(data_set)), key='pupil')

# x = epochs_p_bl10.columns
# df['long_bl'] = np.NaN
# for s in df['subject_id'].unique():
#     for b in df['block_id'].unique():
#         if ((s==24) & (b==2)) | ((s==24) & (b==3)):
#             df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['trial_nr']<102), 'long_bl'] = np.atleast_2d(epochs_p_bl10.loc[(epochs_p_bl10.index.get_level_values('subject_id')==str(s)) & (epochs_p_bl10.index.get_level_values('block_id')==str(b+1)) & (epochs_p_bl10.index.get_level_values('trial_nr')==1), (x>2)&(x<9)].mean(axis=1).values).T
#             df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['trial_nr']>102) & (df['trial_nr']<203), 'long_bl'] = np.atleast_2d(epochs_p_bl10.loc[(epochs_p_bl10.index.get_level_values('subject_id')==str(s)) & (epochs_p_bl10.index.get_level_values('block_id')==str(b+1)) & (epochs_p_bl10.index.get_level_values('trial_nr')==102), (x>2)&(x<9)].mean(axis=1).values).T
#         elif ((s==5) & (b==1)) | ((s==5) & (b==2)):
#             df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['trial_nr']<102), 'long_bl'] = np.atleast_2d(epochs_p_bl10.loc[(epochs_p_bl10.index.get_level_values('subject_id')==str(s)) & (epochs_p_bl10.index.get_level_values('block_id')==str(b+1)) & (epochs_p_bl10.index.get_level_values('trial_nr')==1), (x>2)&(x<9)].mean(axis=1).values).T
#             df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['trial_nr']>102) & (df['trial_nr']<203), 'long_bl'] = np.atleast_2d(epochs_p_bl10.loc[(epochs_p_bl10.index.get_level_values('subject_id')==str(s)) & (epochs_p_bl10.index.get_level_values('block_id')==str(b+1)) & (epochs_p_bl10.index.get_level_values('trial_nr')==102), (x>2)&(x<9)].mean(axis=1).values).T
#             df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['trial_nr']>203) & (df['trial_nr']<304), 'long_bl'] = np.atleast_2d(epochs_p_bl10.loc[(epochs_p_bl10.index.get_level_values('subject_id')==str(s)) & (epochs_p_bl10.index.get_level_values('block_id')==str(b+1)) & (epochs_p_bl10.index.get_level_values('trial_nr')==203), (x>2)&(x<9)].mean(axis=1).values).T
#         else:
#             df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['trial_nr']<102), 'long_bl'] = np.atleast_2d(epochs_p_bl10.loc[(epochs_p_bl10.index.get_level_values('subject_id')==str(s)) & (epochs_p_bl10.index.get_level_values('block_id')==str(b+1)) & (epochs_p_bl10.index.get_level_values('trial_nr')==1), (x>2)&(x<9)].mean(axis=1).values).T
#             df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['trial_nr']>102) & (df['trial_nr']<203), 'long_bl'] = np.atleast_2d(epochs_p_bl10.loc[(epochs_p_bl10.index.get_level_values('subject_id')==str(s)) & (epochs_p_bl10.index.get_level_values('block_id')==str(b+1)) & (epochs_p_bl10.index.get_level_values('trial_nr')==102), (x>2)&(x<9)].mean(axis=1).values).T
#             df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['trial_nr']>203) & (df['trial_nr']<304), 'long_bl'] = np.atleast_2d(epochs_p_bl10.loc[(epochs_p_bl10.index.get_level_values('subject_id')==str(s)) & (epochs_p_bl10.index.get_level_values('block_id')==str(b+1)) & (epochs_p_bl10.index.get_level_values('trial_nr')==203), (x>2)&(x<9)].mean(axis=1).values).T
#             df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['trial_nr']>304) & (df['trial_nr']<405), 'long_bl'] = np.atleast_2d(epochs_p_bl10.loc[(epochs_p_bl10.index.get_level_values('subject_id')==str(s)) & (epochs_p_bl10.index.get_level_values('block_id')==str(b+1)) & (epochs_p_bl10.index.get_level_values('trial_nr')==304), (x>2)&(x<9)].mean(axis=1).values).T

# bins=[0, 102, 203, 304, 405]
# bin_names=[1,2,3,4]
# df['long_bl_phase'] = pd.cut(df['trial_nr'], bins=bins, labels=bin_names)

# ## long baseline interaction
# df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'cond_SOA', 'long_bl_phase'])
# df_res0=df_res.loc[df_res['cond_SOA']==0].reset_index()
# df_res1=df_res.loc[df_res['cond_SOA']!=0].reset_index()
# for s in df_res1['subject_id'].unique():
#     for b in df_res1['long_bl_phase'].unique():
#         df_res1.loc[(df_res1['subject_id']==s) & (df_res1['long_bl_phase']==b),'c_abs_diff'] = df_res1.loc[(df_res1['subject_id']==s) & (df_res1['long_bl_phase']==b),'c_abs'] - df_res0.loc[(df_res0['subject_id']==s) & (df_res0['long_bl_phase']==b),'c_abs'].values

# for m,n,y,z in zip(['pupil_r_diff', 'c_abs_diff'], [df_res, df_res1],[0.8,-0.095],[[0.2,2.75],[-0.15,0.085]]): ## define m
#     fig = plt.figure(figsize=(2.3,2.3))
#     ax = fig.add_subplot() #121)

#     if m == 'pupil_r_diff':
#         ## model for cond_SOA, baseline and interaction effects in AS pupil response or sdt metrics
#         model = AnovaRM(data=df_res.loc[df_res['cond_SOA']!=0], depvar=m, subject='subject_id', within=['cond_SOA', 'long_bl_phase'])
#         res = model.fit()

#         ## plots for cond_SOA, baseline and interaction effects in AS pupil response or sdt metrics
#         # sns.pointplot(data=df_res.loc[df_res['cond_SOA']!=0], x='long_bl_phase', y=m, hue='cond_SOA', palette = [sns.color_palette(color_blend,3)[0],sns.color_palette(color_blend,3)[1],sns.color_palette(color_blend,3)[2]]) 
#         # sns.lineplot(data=df_res.loc[df_res['cond_SOA']!=0], x='long_bl_phase', y=m, hue='cond_SOA', errorbar='se', palette = [sns.color_palette(color_blend,4)[0],sns.color_palette(color_blend,4)[1],sns.color_palette(color_blend,4)[2],sns.color_palette(color_blend,4)[3]]) 
#         means = df_res.groupby(['cond_SOA','long_bl_phase']).mean()
#         sems = df_res.groupby(['cond_SOA','long_bl_phase']).sem()
#         for (i, means_n), (j, sems_n) in zip(means.groupby('cond_SOA'), sems.groupby('cond_SOA')):
#             if i!=0:
#                 plt.errorbar(x=means_n['long_bl'], y=means_n[m], xerr=sems_n['long_bl'], yerr=sems_n[m], fmt='o', color=sns.color_palette(color_blend,4)[i-1], alpha=0.7)

#         plt.ylabel('Δ Pupil response (% change)')
#         coefs = ['SOA:           ', 'Pupil:          ', 'Interaction: ']
#         for i in range(3):
#             coef = coefs[i]
#             f = res.anova_table["F Value"][i] #FIXME so far this is a 2-factor anova, here we're accessing the cond_SOA coefficient which is not complete and not the one of interest
#             df1 = res.anova_table["Num DF"][i]
#             df2 = res.anova_table["Den DF"][i]
#             s = res.anova_table["Pr > F"][i]
#             if s < 0.0001:
#                 txt = ', p < 0.001'
#             else:
#                 txt = ', p = {}'.format(round(s,3))
#             plt.text(x=-5, y=y-(i*0.23), s='{}'.format(coef)+'$F_{{({},{})}}=$'.format(df1, df2)+str(round(f,3))+txt, size=6, va='center', ha='left') # , c=color_AS)
    
#     if m == 'c_abs_diff':
#         ## model for cond_SOA, baseline and interaction effects in psychometric function
#         model1 = AnovaRM(data=df_res1, depvar=m, subject='subject_id', within=['cond_SOA', 'long_bl_phase'])
#         res1 = model1.fit()

#         ##plot for cond_SOA, baseline and interaction effects in psychometric function
#         # sns.lineplot(data=df_res1, x='long_bl_phase', y=m, hue='cond_SOA', errorbar='se', palette = [sns.color_palette(color_blend,4)[0],sns.color_palette(color_blend,4)[1],sns.color_palette(color_blend,4)[2],sns.color_palette(color_blend,4)[3]]) 
#         means = df_res1.groupby(['cond_SOA','long_bl_phase']).mean()
#         sems = df_res1.groupby(['cond_SOA','long_bl_phase']).sem()
#         for (i, means_n), (j, sems_n) in zip(means.groupby('cond_SOA'), sems.groupby('cond_SOA')):
#             if i!=0:
#                 plt.errorbar(x=means_n['long_bl'], y=means_n[m], xerr=sems_n['long_bl'], yerr=sems_n[m], fmt='o', color=sns.color_palette(color_blend,4)[i-1], alpha=0.7)

#         coefs = ['SOA:           ', 'Pupil:          ', 'Interaction: ']
#         for i in range(3):
#             coef = coefs[i]
#             f = res1.anova_table["F Value"][i] #FIXME so far this is a 2-factor anova, here we're accessing the cond_SOA coefficient which is not complete and not the one of interest
#             df1 = res1.anova_table["Num DF"][i]
#             df2 = res1.anova_table["Den DF"][i]
#             s = res1.anova_table["Pr > F"][i]
#             if s < 0.0001:
#                 txt = ', p < 0.001'
#             else:
#                 txt = ', p = {}'.format(round(s,3))
#             plt.text(x=-5, y=y-0.02*i, s= '{}'.format(coef)+'$F_{{({},{})}}=$'.format(df1, df2)+str(round(f,3))+txt, size=6, va='center', ha='left') # , c=sns.color_palette(color_blend,3)[i])
#         plt.ylabel('Δ | Criterion |')

#     plt.ylim(z[0],z[1])
#     plt.xlabel('Pupil response (% change)')
#     # plt.xticks([0,1,2])
#     plt.legend([], [], frameon=False)
#     sns.despine(trim=True)
#     plt.tight_layout()
#     # fig.savefig(os.path.join(figs_dir, '{}_{}_on_long_baseline.pdf'.format(data_set, m)))

# ## one way ANOVA
# df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'pupil_b_bin'])
# df_res1=df_res.loc[df_res['condition']==1].reset_index()
# df_res2=df_res.loc[df_res['condition']==0].reset_index()
# df_res1['c_abs2']=df_res2['c_abs']
# df_res1['c_abs_diff']=df_res1['c_abs']-df_res1['c_abs2']

# for m,n,y,z in zip(['c_abs_diff', 'pupil_r_diff'], [df_res1, df_res],[-0.11,-1.3], [[-0.12,0.06],[-1.6,4.5]]):
#     n = n.groupby(['subject_id', 'condition', 'pupil_b_bin']).mean().reset_index()
#     model = AnovaRM(data=n.loc[n['condition']==1], depvar=m, subject='subject_id', within=['pupil_b_bin'])

#     # model = AnovaRM(data=n, depvar=m, subject='subject_id', within=['pupil_b_bin'])
#     res = model.fit()
#     print(model.fit())

#     fig = plt.figure(figsize=(2,2))
#     ax = fig.add_subplot() #121)
#     sns.lineplot(data=n, x='pupil_b_bin', y=m, errorbar='se', palette = [color_AS])
#     # sns.lineplot(data=df_res, x='pupil_b_bin', y='c_abs', hue='condition', errorbar='se', palette = [color_noAS, color_AS]) 
    
#     f = res.anova_table["F Value"][0]
#     df1 = res.anova_table["Num DF"][0]
#     df2 = res.anova_table["Den DF"][0]
#     s = res.anova_table["Pr > F"][0]
#     if s < 0.0001:
#         txt = ', p < 0.001'
#     else:
#         txt = ', p = {}'.format(round(s,3))
#     plt.text(x=1, y=y, s='$F_{{({},{})}}=$'.format(df1, df2)+str(round(f,3))+txt, size=6, va='center', ha='center') # , c=color_AS)
#     plt.ylim(z[0],z[1])
#     if m == 'pupil_r_diff':
#         plt.ylabel('Δ Pupil response (% change)')
#     if m == 'c_abs_diff':
#         ax.set_ylabel('Δ | Criterion c |')
#     plt.xlabel('Pupil baseline bin')
#     plt.xticks([0,1,2])
#     plt.legend([], [], frameon=False)

#     sns.despine(trim=True)
#     plt.tight_layout()
#     # fig.savefig(os.path.join(figs_dir, '{}_{}_on_baseline_bin.pdf'.format(data_set, m)))

# ## Pupil response to AS across trials older plot with title
# ## task-irrelevant stimulus-evoked pupil response across trials
# n_jobs = 4
# n_boot = 100
# means_as = []
# sems_as = []

# def get_tis_resp(epochs, groupby, maxi, iteration):
#     as_resp = pd.DataFrame()
#     y = np.array(epochs.columns, dtype=float)
#     as_resp['p'] = epochs.loc[:,(y>0)&(y<maxi)].groupby(groupby).mean().max(axis=1)
#     as_resp['iteration'] = iteration
#     return as_resp
# as_resp = get_tis_resp(epochs=epochs_diff_sound.loc[epochs_diff_sound.index.get_level_values(3)=='boost'], groupby='trial_nr', maxi=2, iteration=0)
# means_as.append(as_resp)
# sem_as_ = pd.concat(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(get_tis_resp)
#                                             (epochs_diff_sound.loc[epochs_diff_sound.index.get_level_values(3)=='boost'].sample(frac=1, replace=True), ['trial_nr'], 2, i)
#                                             for i in range(n_boot)))
# sems_as.append((sem_as_.groupby(['trial_nr']).quantile(0.84) - sem_as_.groupby(['trial_nr']).quantile(0.16))/2)

# fig = utils_hh_contrast_detection.plot_p_across_trials(means_as, sems_as, y='p')
# # fig.savefig(os.path.join(figs_dir, '{}_across_trials_as_p_resp.pdf'.format(data_set, y)))
