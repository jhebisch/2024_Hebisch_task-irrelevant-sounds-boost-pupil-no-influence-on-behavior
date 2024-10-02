'''
Second and final script to be applied for analysis of Experiment 3.
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statistics

import utils
import utils_amsterdam_orientation_discrimination

def zscore_dvs(df):
    mean = df[['dv0', 'dv1', 'dv2', 'dv3', 'dv4', 'dv5', 'dv6', 'dv7']].values.ravel().mean()
    std = df[['dv0', 'dv1', 'dv2', 'dv3', 'dv4', 'dv5', 'dv6', 'dv7']].values.ravel().std()
    for i in range(8):
        df['dv{}'.format(i)] = (df['dv{}'.format(i)]-mean)/std
    return df

def compute_psychometric_function(df):
    import statsmodels.formula.api as smf
    model = smf.logit('choice ~ dv', data=df).fit()
    return model.params # , model

def compute_kernel(df, formula='choice ~ 1 + dv0 + dv1 + dv2 + dv3 + dv4 + dv5 + dv6 + dv7'):
    import statsmodels.formula.api as smf
    model = smf.logit(formula, data=df).fit()
    return model.params

def compute_kernel2(df, formula='choice ~ dv0_n + dv1_n + dv2_n + dv3_n + dv4_n + dv5_n + dv6_n + dv7_n'): # def compute_kernel(df, formula='choice ~ dv0 + dv1 + dv2 + dv3 + dv4 + dv5 + dv6 + dv7'):
    import statsmodels.formula.api as smf
    model = smf.logit(formula, data=df).fit()
    return model.params

def compute_kernel_bl(df, formula='choice ~ dv0_n + dv1_n + dv2_n + dv3_n + dv4_n + dv5_n + dv6_n + dv7_n + pupil_b_bin'): # def compute_kernel(df, formula='choice ~ dv0 + dv1 + dv2 + dv3 + dv4 + dv5 + dv6 + dv7'):
    import statsmodels.formula.api as smf
    model = smf.logit(formula, data=df).fit()
    return model.params


def compute_r(df):
    x1, x0 = np.polyfit(x=np.arange(df.shape[1]), y=df.iloc[0].values, deg=1)
    return pd.DataFrame({'x0':[x0], 'x1':[x1]})

def plot_psycho_kernel_logit(df):

    colors = {0: color_noAS,
              1: sns.color_palette(color_blend,3)[0],
              2: sns.color_palette(color_blend,3)[1],
              3: sns.color_palette(color_blend,3)[2]}

    psycho = df.groupby(['subject_id', 'condition']).apply(compute_kernel)
    psycho = psycho.drop(['Intercept'], axis=1)

    # plot:
    fig = plt.figure(figsize=(2,2))
    x = np.arange(psycho.shape[1])+1
    for cond in [0,1,2,3]:
        
        data = psycho.query('condition == {}'.format(cond))
        plt.errorbar(x, data.mean(), yerr=data.sem(), fmt='-o', label=cond, color=colors[cond], alpha=0.7)
        if cond == 0:
            data0 = psycho.query('condition == {}'.format(cond))
        else:
            p_values = [sp.stats.wilcoxon(data.iloc[:,dv], data0.iloc[:,dv])[1] for dv in range(8)]
            print(p_values)
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.xlabel('Stimulus #')
    plt.ylabel('Weight')
    # plt.legend(loc=2)
    sns.despine(trim=True)
    plt.tight_layout()

    return fig

def plot_psycho_regression(df):

    colors = {0: sns.color_palette()[0],
              1: sns.color_palette()[1],
              2: sns.color_palette()[2],
              3: sns.color_palette()[3]}

    formula = 'choice ~ dv0 + dv1 + dv2 + dv3 + dv4 + dv5 + dv6 + dv7'
    psycho = df.groupby(['subject_id', 'condition']).apply(compute_kernel, formula=formula)
    psycho = psycho.drop(['Intercept'], axis=1)

    coefs = psycho.groupby(['subject_id', 'condition']).apply(compute_r).reset_index()

    p_values = []
    for m in ['x0', 'x1']:
        for c in [1, 2, 3]:
            p_values.append(sp.stats.ttest_rel(coefs.loc[coefs['condition']==c, m],
                            coefs.loc[coefs['condition']==0, m])[1])

    fig = plt.figure(figsize=(4,2))
    ax = fig.add_subplot(121)
    x = np.arange(psycho.shape[1])
    for cond in [0,1,2,3]:
        data = coefs.query('condition == {}'.format(cond))
        lines = []
        for i in range(data.shape[0]):
            lines.append(x*data['x1'].iloc[i] + data['x0'].iloc[i])
        lines = np.vstack(lines)
        mean = np.mean(lines, axis=0)
        sem = sp.stats.sem(lines, axis=0)
        plt.fill_between(x, mean-sem, mean+sem, color=colors[cond], alpha=0.1)
        plt.plot(x, mean, color=colors[cond], label=cond)
    plt.xlabel('Stim #')
    plt.ylabel('Weight')
    plt.legend(loc=2)

    ax = fig.add_subplot(122)
    for j, cond in enumerate([0,1,2,3]):
        data = coefs.query('condition == {}'.format(cond))
        plt.errorbar(x=np.array([j]), y=data['x1'].mean(), yerr=data['x1'].sem(), fmt='-o', color=colors[cond])
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for j, cond in enumerate([1,2,3]):
        plt.text(x=j+1, y=0.9, s=str(round(p_values[j+3],3)), size=6, transform=trans)
    plt.xticks([0,1,2,3], ['normal', 'AS_0', 'AS_1', 'AS_2'])
    plt.ylabel('Slope')

    sns.despine(trim=True)
    plt.tight_layout()
    
    return fig

def plot_bias_measures(df):

    formula = 'choice ~ dv0 + dv1 + dv2 + dv3 + dv4 + dv5 + dv6 + dv7 + stimulus_p + choice_p'
    coefs = df.groupby(['subject_id', 'condition']).apply(compute_kernel, formula=formula)
    coefs = coefs[['Intercept', 'stimulus_p', 'choice_p']].reset_index()

    coefs[['Intercept_abs', 'stimulus_p_abs', 'choice_p_abs']] = coefs[['Intercept', 'stimulus_p', 'choice_p']].abs()

    colors = {0: sns.color_palette()[0],
                1: sns.color_palette()[1],
                2: sns.color_palette()[2],
                3: sns.color_palette()[3]}

    fig = plt.figure(figsize=(6,4))

    plt_nr = 1
    for m in ['Intercept', 'stimulus_p', 'choice_p',
            'Intercept_abs', 'stimulus_p_abs', 'choice_p_abs']:

        p_values = []
        for c in [1,2,3]:
            p_values.append(sp.stats.ttest_rel(coefs.loc[coefs['condition']==c, m],
                            coefs.loc[coefs['condition']==0, m])[1])

        ax = fig.add_subplot(2,3,plt_nr)
        for j, cond in enumerate([0,1,2,3]):
            data = coefs.query('condition == {}'.format(cond))
            plt.errorbar(x=np.array([j]), y=data[m].mean(), yerr=data[m].sem(), fmt='-o', color=colors[cond])
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for j, cond in enumerate([1,2,3]):
            plt.text(x=j+1, y=0.9, s=str(round(p_values[j],3)), size=6, transform=trans)
        plt.xticks([0,1,2,3], ['normal', 'AS_0', 'AS_1', 'AS_2'])
        plt.ylabel(m)
        plt_nr += 1
    sns.despine(trim=True)
    plt.tight_layout()

    return fig

def plot_psycho_kernel2(df, normalize=False):

    # first pass:
    means_c = df.loc[(df['response']==0),:].groupby(['subject_id', 'condition'])[['DV_   {}'.format(i) for i in range(8)]].mean()
    means_d = df.loc[(df['response']==1),:].groupby(['subject_id', 'condition'])[['DV_   {}'.format(i) for i in range(8)]].mean()
    psycho = means_d - means_c

    # normalize:
    if normalize:
        means_c_stim = df.loc[(df['stimulus']==0),:].groupby(['subject_id'])[['DV_   {}'.format(i) for i in range(8)]].mean()
        means_d_stim = df.loc[(df['stimulus']==1),:].groupby(['subject_id'])[['DV_   {}'.format(i) for i in range(8)]].mean()
        psycho_stim = (means_d_stim-means_c_stim).mean(axis=1)
        for i in range(psycho.shape[0]):
            subj = psycho.index.get_level_values('subject_id')[i]
            psycho.iloc[i] = psycho.iloc[i] / psycho_stim.loc[subj]

    # difference:
    psycho_normal = psycho.loc[psycho.index.get_level_values('condition')=='normal']
    psycho_diff = psycho.loc[psycho.index.get_level_values('condition')!='normal']
    for i in range(psycho_diff.shape[0]):
        subj = psycho_diff.index.get_level_values('subject_id')[i]
        psycho_diff.iloc[i] = psycho_diff.iloc[i] - psycho_normal.loc[subj,:]

    # plot:
    fig = plt.figure(figsize=(12,3))
    plt_nr = 1
    for cond in ['normal', 'AS_0', 'AS_1', 'AS_2']:
        
        p = psycho.loc[psycho.index.get_level_values('condition')==cond,:]

        ax = fig.add_subplot(1,4,plt_nr)
        x = np.arange(8)
        for i in range(p.shape[0]):
            plt.plot(x, p.iloc[i])
        # plt.ylim(0.3,1)
        plt.xlabel('Stim #')
        plt.ylabel('Weight')
        plt.title(cond)
        plt_nr += 1

    sns.despine(trim=True)
    plt.tight_layout()

    return fig

def compute_results(df, groupby, iteration=0):

    # groupby = ['subject_id', 'session_id', 'condition']
    df_res1 = df.groupby(groupby).mean()
    df_res1['choice_abs'] = abs(df_res1['choice']-0.5)
    df_res2 = df.groupby(groupby).apply(utils.sdt, stim_column='stimulus', response_column='choice')
    df_res = reduce(lambda left, right: pd.merge(left,right, on=groupby), [df_res1, df_res2]).reset_index()
    df_res['c_abs'] = abs(df_res['c'])
    df_res['strength'] = 0

    # df_res_d = (df_res.loc[df_res['condition']==1,:].set_index(groupby).droplevel(level=1) - 
    #             df_res.loc[df_res['condition']==0,:].set_index(groupby).droplevel(level=1)).reset_index()
    # df_res_o = ((df_res.loc[df_res['condition']==1,:].set_index(groupby).droplevel(level=1) + 
    #             df_res.loc[df_res['condition']==0,:].set_index(groupby).droplevel(level=1))/2).reset_index()
    
    df_res['iteration'] = iteration
    # df_res_o['iteration'] = iteration
    # df_res_d['iteration'] = iteration

    return df_res

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
    'ytick.color':'Black',} 
    )
sns.plotting_context()
color_noAS = '#DECA59' #'#CBC034' 
color_AS = '#6CA6F0'
color_blend = "blend:#6CA6F0,#702D96"
sns.set_palette(color_blend, n_colors=3)

# define folders
# project_dir = os.path.dirname(os.getcwd())
# data_dir = os.path.join(project_dir, 'data')
# figs_dir = os.path.join(project_dir, 'figs')
project_dir = 'G:\\.shortcut-targets-by-id\\1-KxMllxLjzCUetyGI01Gp0sVPupwLl38\\2023_as_multiple_datasets' # os.path.dirname(os.getcwd()) #TODO change back in the end
data_dir = os.path.join(project_dir, 'data')
figs_dir = os.path.join(project_dir, 'figs') #, 'color suggestion') #TODO change back when color scheme was decided upon
data_set = 'amsterdam_orientation_discrimination'
trial_cutoff = 9

# load data:
print('loading... {}'.format(data_set))
df = pd.read_csv(os.path.join(data_dir, '{}_df.csv'.format(data_set)))
epochs_p_stim = pd.read_hdf(os.path.join(data_dir, '{}_epochs_p_stim.hdf'.format(data_set)), key='pupil')
epochs_p_resp = pd.read_hdf(os.path.join(data_dir, '{}_epochs_p_resp.hdf'.format(data_set)), key='pupil')

# exclude trials:
exclude = np.array( 
                    (df['correct'] == -1) |
                    (df['rt']>4.5) |
                    (df['blinks_dec']>.2)
                    )
df = df.loc[~exclude,:].reset_index(drop=True)
epochs_p_stim = epochs_p_stim.loc[~exclude,:]
epochs_p_resp = epochs_p_resp.loc[~exclude,:]

# exclude subjects if their trial number falls under 1000:
print('# subjects before = {}'.format(len(df['subject_id'].unique())))
print(df['subject_id'].unique())

counts = df.groupby(['subject_id'])['trial_nr'].count().reset_index()
ind = df['subject_id'].isin(counts.loc[counts['trial_nr']>1000,'subject_id']).values
df = df.loc[ind,:].reset_index(drop=True)
epochs_p_stim = epochs_p_stim.loc[ind,:]
epochs_p_resp = epochs_p_resp.loc[ind,:]
print('# subjects after exclusion = {}'.format(len(df['subject_id'].unique())))
print(df['subject_id'].unique())

# add DV bins:
q = 11
df['dv_bin'] = df.groupby(['subject_id', 'block_id', 'condition'])['dv'].apply(pd.qcut, q=q, labels=False)
df['dv_abs_bin'] = df.groupby(['subject_id', 'block_id', 'condition'])['dv_abs'].apply(pd.qcut, q=q, labels=False)

# create normalised dvs
def matrix_sd(df):
    df2 = df[['dv0', 'dv1', 'dv2','dv3', 'dv4', 'dv5', 'dv6', 'dv7']].stack()
    sd = len(df) * [df2.std()]
    df['dv_sd'] = sd
    return df
## add dv sd to df
df = df.groupby(['subject_id', 'block_id'], sort=False).apply(matrix_sd)
## create dv_ns (normalised dvs)
for i in ['dv0', 'dv1', 'dv2','dv3', 'dv4', 'dv5', 'dv6', 'dv7']:
    j = i + '_n'
    df[j] = df[i] / df['dv_sd']

# add sequential info:
df = df.groupby(['subject_id', 'block_id']).apply(utils.compute_sequential).reset_index(drop=True)
df.loc[df['stimulus_p']==0, 'stimulus_p'] = -1
df.loc[df['choice_p']==0, 'choice_p'] = -1

# add pupil:
## pre-trial baseline (bins)
x = np.array(epochs_p_stim.columns, dtype=float)
df['pupil_b'] = epochs_p_stim.loc[:,(x>-1.5)&(x<-1)].mean(axis=1).values

x = np.array(epochs_p_stim.columns, dtype=float)
df['pupil_tpr_b'] = epochs_p_stim.loc[:,(x>-0.5)&(x<0)].mean(axis=1).values

# make pupil baseline bins
bins = 3
df['pupil_b_bin'] = df.groupby(['subject_id', 'block_id'])['pupil_b'].apply(pd.qcut, q=bins, labels=False)

def compute_pupil_response_diff(epochs_p_stim):

    # baseline:
    x = epochs_p_stim.columns
    epochs_p_stim = epochs_p_stim - np.atleast_2d(epochs_p_stim.loc[:,(x>-1.5)&(x<-1)].mean(axis=1).values).T
    
    # compute mean on normal trials:
    mean = epochs_p_stim.query('condition == "normal"').mean()

    # subtract from each trial:
    epochs_p_stim = epochs_p_stim - mean
    
    # means = epochs_p_stim.groupby(['condition']).mean()
    scalars = np.repeat(0, epochs_p_stim.shape[0])
    ind1 = np.array(epochs_p_stim.index.get_level_values('condition')=='AS_0')
    scalars[ind1] = epochs_p_stim.loc[ind1, (x>-1)&(x<1)].mean(axis=1)
    ind2 = np.array(epochs_p_stim.index.get_level_values('condition')=='AS_1')
    scalars[ind2] = epochs_p_stim.loc[ind2, (x>0)&(x<2)].mean(axis=1)
    ind3 = np.array(epochs_p_stim.index.get_level_values('condition')=='AS_2')
    scalars[ind3] = epochs_p_stim.loc[ind3, (x>1)&(x<3)].mean(axis=1)
    
    return pd.DataFrame({'pupil_r_diff': scalars,})


res = epochs_p_stim.groupby(['subject_id', 'block_id'], sort=False).apply(compute_pupil_response_diff)
df['pupil_r_diff'] = res['pupil_r_diff'].values


x = np.array(epochs_p_resp.columns, dtype=float) 
df['tpr'] = epochs_p_resp.loc[:,(x>-0.5)&(x<0.5)].mean(axis=1).values - df['pupil_tpr_b'] # (x>1.5)&(x<2.5) what sort of an interval is this? #this baseline?

# correct phasic pupil responses and create 8 bins: 
def bin_correct_tpr(df, plus_correct='pupil_tpr_b + rt + dv_abs', trial_cutoff=19):
    import statsmodels.formula.api as smf
    model = smf.ols(formula='tpr ~ {}'.format(plus_correct), data=df.loc[df['trial_nr']>trial_cutoff,:]).fit()
    df.loc[df['trial_nr']>trial_cutoff, 'tpr_c'] = model.resid + df.loc[df['trial_nr']>trial_cutoff, 'tpr'].mean()

    return df

df = df.groupby(['subject_id', 'block_id', 'condition'], group_keys=False).apply(bin_correct_tpr)
bins = 8
df['tpr_bin'] = df.groupby(['subject_id', 'block_id', 'condition'])['tpr_c'].apply(pd.qcut, q=bins, labels=False)
df['pupil_r_diff_bin'] = -1 # np.NaN
# df.loc[df['condition']!=0,'pupil_r_diff_bin'] = df.loc[df['condition']!=0].groupby(['subject_id', 'block_id'])['pupil_r_diff'].apply(pd.qcut, q=bins, labels=False)
for s in df['subject_id'].unique():
    for b in df.loc[(df['subject_id']==s),'block_id'].unique():
        if (s==18) & (b==7):
            df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff_bin'] = pd.cut(df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff'], bins=[-13, -6, -2.25, 0, 3, 6, 10, 21], labels=["0","1","2","4","5","6","7"]) # , labels=False) # , labels=[0,1,2,4,5,6,7])
            df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0) & (df['pupil_r_diff_bin'].notna()),'pupil_r_diff_bin'] = df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0) &  (df['pupil_r_diff_bin'].notna()),'pupil_r_diff_bin'].astype(int)
            print(df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff_bin'].unique())
        elif (s==19) & (b==5):
            df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff_bin'] = pd.cut(df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff'], bins=[-24, -3, -1, 0, 2, 4, 6, 13], labels=["0","1","3","4","5","6","7"])
            df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0) & (df['pupil_r_diff_bin'].notna()),'pupil_r_diff_bin'] = df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0) &  (df['pupil_r_diff_bin'].notna()),'pupil_r_diff_bin'].astype(int)
            print(df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff_bin'].unique())
        elif (s==19) & (b==6):
            df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff_bin'] = pd.cut(df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff'], bins=[-8, -3, -1, 0, 2, 4, 7, 19], labels=["0","1","2","4","5","6","7"])
            df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0) & (df['pupil_r_diff_bin'].notna()),'pupil_r_diff_bin'] = df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0) &  (df['pupil_r_diff_bin'].notna()),'pupil_r_diff_bin'].astype(int)
            print(df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff_bin'].unique())
        elif (s==4) & (b==6):
            df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff_bin'] = pd.cut(df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff'], bins=[-34, -7.25, -3, 0, 3, 7, 14.25, 33], labels=["0","1","3","4","5","6","7"])
            df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0) & (df['pupil_r_diff_bin'].notna()),'pupil_r_diff_bin'] = df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0) &  (df['pupil_r_diff_bin'].notna()),'pupil_r_diff_bin'].astype(int)
            print(df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff_bin'].unique())
        else:
            df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff_bin'] = pd.qcut(df.loc[(df['subject_id']==s) & (df['block_id']==b) & (df['condition']!=0),'pupil_r_diff'], q=bins, labels=False)

# baseline:
epochs_p_stim = epochs_p_stim - np.atleast_2d(df['pupil_b'].values).T
epochs_p_resp = epochs_p_resp - np.atleast_2d(df['pupil_b'].values).T

# mean behaviour 
pf = df.loc[df['trial_nr']>trial_cutoff,:].groupby(['subject_id']).apply(compute_kernel2).reset_index()
pf['intercept_abs'] = abs(pf['Intercept'])
pf['dv'] = (pf['dv0_n'] +pf['dv1_n'] + pf['dv2_n'] + pf['dv3_n'] + pf['dv4_n'] + pf['dv5_n'] + pf['dv6_n'] + pf['dv7_n'])/8
pf['dv_abs'] = abs(pf['dv'])
df_res = compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id'])

m_shift = pf.Intercept.mean()
sem_shift = pf.Intercept.sem()
m_dv = pf.dv.mean()
sem_dv = pf.dv.sem()
m_rt = df_res.rt.mean()
sem_rt = df_res.rt.sem()
m_correct = df_res.correct.mean()
sem_correct = df_res.correct.sem()

###################### Plots and analyses ##########################

# ------------------------------------------------------------------
# Fig. 4 Task-irrelevant stimulus evoked pupil response
# ------------------------------------------------------------------

## pupil response time courses
x = epochs_p_stim.columns
fig = utils_amsterdam_orientation_discrimination.plot_pupil_responses_amsterdam_orientation(epochs_p_stim.query('trial_nr > {}'.format(trial_cutoff)).loc[:, (x>-1)&(x<4)])
# fig.savefig(os.path.join(figs_dir, '{}_pupil_responses.pdf'.format(data_set)))

## pupil response stats
df_res = compute_results(df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition'])
m = 'pupil_r_diff'
fig = plt.figure(figsize=(2,2)) #2.5))
ax = fig.add_subplot() #121)
y = df_res.loc[df_res['condition']==1, m].values-df_res.loc[df_res['condition']==0, m].values
sns.stripplot(x=1, y=y, 
            color=sns.color_palette(color_blend,3)[0], ax=ax, linewidth=0.2, alpha=0.7, edgecolor=sns.color_palette(color_blend,3)[0])
ax.hlines(y=y.mean(), xmin=(0)-0.4, xmax=(0)+0.4, zorder=10, colors='k')
y = df_res.loc[df_res['condition']==2, m].values-df_res.loc[df_res['condition']==0, m].values
sns.stripplot(x=2, y=y, 
            color=sns.color_palette(color_blend,3)[1], ax=ax, linewidth=0.2, alpha=0.7, edgecolor=sns.color_palette(color_blend,3)[1])
ax.hlines(y=y.mean(), xmin=(1)-0.4, xmax=(1)+0.4, zorder=10, colors='k')
df_res.loc[df_res['condition']==3, m].values-df_res.loc[df_res['condition']==0, m].values
sns.stripplot(x=3, y=y, 
            color=sns.color_palette(color_blend,3)[2], ax=ax, linewidth=0.2, alpha=0.7, edgecolor=sns.color_palette(color_blend,3)[2])
ax.hlines(y=y.mean(), xmin=(2)-0.4, xmax=(2)+0.4, zorder=10, colors='k')
p_values = []
bf_values = []
for c in [1,2,3]:
    p_values.append(sp.stats.ttest_rel(df_res.loc[df_res['condition']==c, m].values,
                    df_res.loc[df_res['condition']==0, m].values)[1])
    t_value = sp.stats.ttest_rel(df_res.loc[df_res['condition']==c, m].values,
                    df_res.loc[df_res['condition']==0, m].values)[0]
    bf_values.append(bayesfactor_ttest(t_value, len(df['subject_id'].unique()), paired=True))
    print(data_set, m, 'p,t,bf for condition', c, ':', p_values[-1], t_value, '{0: .3g},'.format(round(bf_values[-1]), 3))
p_values = [p * 3 for p in p_values]
model_anova = AnovaRM(data=df_res, depvar=m, subject='subject_id', within=['condition'])
print(model_anova.fit())
plt.axhline(0, color='black', lw=0.5)
if m == 'pupil_r_diff':
    plt.ylabel('Δ Pupil response (% change)')
plt.xlabel('Time from 1st stimulus (s)')
ax.set_xticklabels([-1, 0, 1])
sns.despine(trim=True)
plt.tight_layout()
plt.subplots_adjust(wspace=0.5)
# fig.savefig(os.path.join(figs_dir, '{}_as_{}.pdf'.format(data_set, m)))

# ------------------------------------------------------------------
# Fig. 4 Task-irrelevant stimulus effects on behaviour
# ------------------------------------------------------------------

## use psychometric function
pf = df.loc[df['trial_nr']>trial_cutoff,:].groupby(['subject_id', 'condition']).apply(compute_kernel2).reset_index() #FIXME run again now with cut-off
pf['intercept_abs'] = abs(pf['Intercept'])
pf['dv'] = (pf['dv0_n'] +pf['dv1_n'] + pf['dv2_n'] + pf['dv3_n'] + pf['dv4_n'] + pf['dv5_n'] + pf['dv6_n'] + pf['dv7_n'])/8
pf['dv_abs'] = abs(pf['dv'])
for m in ['dv', 'Intercept', 'intercept_abs']:
    if m == 'intercept_abs':
        fig = plt.figure(figsize=(2,2))
    else:
        fig = plt.figure(figsize=(2,2.5))
    ax = fig.add_subplot() #121)
    y = pf.loc[pf['condition']==1, m].values-pf.loc[pf['condition']==0, m].values
    sns.stripplot(x=1, y=y, 
                color=sns.color_palette(color_blend,3)[0], ax=ax, linewidth=0.2, alpha=0.7, edgecolor=sns.color_palette(color_blend,3)[0])
    ax.hlines(y=y.mean(), xmin=(0)-0.4, xmax=(0)+0.4, zorder=10, colors='k')
    y = pf.loc[pf['condition']==2, m].values-pf.loc[pf['condition']==0, m].values
    sns.stripplot(x=2, y=y, 
                color=sns.color_palette(color_blend,3)[1], ax=ax, linewidth=0.2, alpha=0.7, edgecolor=sns.color_palette(color_blend,3)[1])
    ax.hlines(y=y.mean(), xmin=(1)-0.4, xmax=(1)+0.4, zorder=10, colors='k')
    pf.loc[pf['condition']==3, m].values-pf.loc[pf['condition']==0, m].values
    sns.stripplot(x=3, y=y, 
                color=sns.color_palette(color_blend,3)[2], ax=ax, linewidth=0.2, alpha=0.7, edgecolor=sns.color_palette(color_blend,3)[2])
    ax.hlines(y=y.mean(), xmin=(2)-0.4, xmax=(2)+0.4, zorder=10, colors='k')
    p_values = []
    bf_values = []
    for c in [1,2,3]:
        p_values.append(sp.stats.ttest_rel(pf.loc[pf['condition']==c, m].values,
                        pf.loc[pf['condition']==0, m].values)[1])
        t_value = sp.stats.ttest_rel(pf.loc[pf['condition']==c, m].values,
                        pf.loc[pf['condition']==0, m].values)[0]
        bf_values.append(bayesfactor_ttest(t_value, len(df['subject_id'].unique()), paired=True))
        print(data_set, m, 'p, t, bf for condition', c, ':', p_values[-1], t_value, '{0: .3g},'.format(round(bf_values[-1]),3))
    p_values = [p * 3 for p in p_values]
    model_anova = AnovaRM(data=pf, depvar=m, subject='subject_id', within=['condition'])
    print(m, 'anova', model_anova.fit())
    print(m, 'BFs', bf_values)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    if (m == 'Intercept') or (m == 'dv'):
        for j, cond in enumerate([1,2,3]):
            if m == 'pupil_r_diff':
                plt.text(x=j, y=0.9, s=str(round(p_values[j],3)), size=6, rotation=45, transform=trans)
            else:
                plt.text(x=j+0.1, y=1, s='$BF_{{{}}}={}$'.format('10',round(bf_values[j],3)), size=5, rotation=90, transform=trans)
    plt.axhline(0, color='black', lw=0.5)
    if m == 'Intercept':
        plt.ylabel('Δ Shift')
    if m == 'intercept_abs':
        plt.ylabel('Δ | Shift |')
    if m == 'dv':
        plt.ylabel('Δ Slope')
    plt.xlabel('Time from 1st stimulus (s)')
    ax.set_xticklabels([-1,0,1])
    sns.despine(trim=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    # fig.savefig(os.path.join(figs_dir, '{}_as_behavior_{}.pdf'.format(data_set, m)))

## use psychometric kernels
df1 = df.loc[df['trial_nr']>trial_cutoff,:].groupby(['subject_id', 'condition']).apply(zscore_dvs).reset_index(drop=True)
fig = plot_psycho_kernel_logit(df1)
# fig.savefig(os.path.join(figs_dir, '{}_kernels.pdf'.format(data_set)))

# ------------------------------------------------------------------
# Fig. 5: baseline pupil size and task-irrelevant stimulus effect interaction 
# ------------------------------------------------------------------
df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'pupil_b_bin'])
## make a dataframe containing differences in c_abs between each AS condition and the no-AS condition
df_res1=df_res.loc[df_res['condition']==1].reset_index()
df_res2=df_res.loc[df_res['condition']==0].reset_index()
df_res3=df_res.loc[df_res['condition']==2].reset_index()
df_res4=df_res.loc[df_res['condition']==3].reset_index()
df_res1['c_abs0']=df_res2['c_abs']
df_res1['c_abs2']=df_res3['c_abs']
df_res1['c_abs3']=df_res4['c_abs']
df_res1['c_abs_diff_1']=df_res1['c_abs']-df_res1['c_abs0']
df_res1['c_abs_diff_2']=df_res1['c_abs2']-df_res1['c_abs0']
df_res1['c_abs_diff_3']=df_res1['c_abs3']-df_res1['c_abs0']

## use psychometric funtion
# pf = compute_kernel2(df.loc[df['trial_nr']>trial_cutoff,:].groupby(['subject_id','condition']), formula='choice ~ dv0_n + dv1_n + dv2_n + dv3_n + dv4_n + dv5_n + dv6_n + dv7_n + pupil_b_bin')
# pf = df.loc[df['trial_nr']>trial_cutoff,:].groupby(['subject_id', 'condition']).apply(compute_kernel_bl).reset_index() #FIXME run again now with cut-off
# pf = compute_kernel2(df.loc[df['trial_nr']>trial_cutoff,:].groupby(['subject_id','condition', 'pupil_b_bin']), formula='choice ~ dv0_n + dv1_n + dv2_n + dv3_n + dv4_n + dv5_n + dv6_n + dv7_n')
pf = df.loc[df['trial_nr']>trial_cutoff,:].groupby(['subject_id', 'condition', 'pupil_b_bin']).apply(compute_kernel2).reset_index() #FIXME run again now with cut-off
pf = pf.reset_index()
pf['intercept_abs'] = abs(pf['Intercept'])
pf['dv'] = (pf['dv0_n'] +pf['dv1_n'] + pf['dv2_n'] + pf['dv3_n'] + pf['dv4_n'] + pf['dv5_n'] + pf['dv6_n'] + pf['dv7_n'])/8
pf['dv_abs'] = abs(pf['dv'])
pf0=pf.loc[pf['condition']==0].reset_index()
pf1=pf.loc[pf['condition']!=0].reset_index()
for s in pf1['subject_id'].unique():
    for b in pf1['pupil_b_bin'].unique():
        pf1.loc[(pf1['subject_id']==s) & (pf1['pupil_b_bin']==b),'int_abs_diff']=pf1.loc[(pf1['subject_id']==s) & (pf1['pupil_b_bin']==b),'intercept_abs']-pf0.loc[(pf0['subject_id']==s) & (pf0['pupil_b_bin']==b),'intercept_abs'].values

# pf1=pf.loc[pf['condition']==1].reset_index()
# pf2=pf.loc[pf['condition']==2].reset_index()
# pf3=pf.loc[pf['condition']==3].reset_index()
# pf0['int_abs1']=pf1['intercept_abs']
# pf0['int_abs2']=pf2['intercept_abs']
# pf0['int_abs3']=pf3['intercept_abs']
# pf0['int_abs_diff_1']=pf0['int_abs1']-pf0['intercept_abs']
# pf0['int_abs_diff_2']=pf0['int_abs2']-pf0['intercept_abs']
# pf0['int_abs_diff_3']=pf0['int_abs3']-pf0['intercept_abs']

for m,n,y,z in zip(['pupil_r_diff', 'int_abs_diff'], [df_res, pf1],[-9,-0.14],[[-15,12.5],[-0.25,0.27]]): ## define m
    fig = plt.figure(figsize=(2.5,2.3))
    ax = fig.add_subplot() #121)

    if m == 'pupil_r_diff':
        ## model for condition, baseline and interaction effects in AS pupil response or sdt metrics
        model = AnovaRM(data=df_res.loc[df_res['condition']!=0], depvar=m, subject='subject_id', within=['condition', 'pupil_b_bin'])
        res = model.fit()

        ## plots for condition, baseline and interaction effects in AS pupil response or sdt metrics
        # sns.pointplot(data=df_res.loc[df_res['condition']!=0], x='pupil_b_bin', y=m, hue='condition', palette = [sns.color_palette(color_blend,3)[0],sns.color_palette(color_blend,3)[1],sns.color_palette(color_blend,3)[2]]) 
        # sns.lineplot(data=df_res.loc[df_res['condition']!=0], x='pupil_b_bin', y=m, hue='condition', errorbar='se', palette = [sns.color_palette(color_blend,3)[0],sns.color_palette(color_blend,3)[1],sns.color_palette(color_blend,3)[2]]) 
        for i in [1,2,3]:
            means = n.loc[n['condition']==i].groupby(['pupil_b_bin']).mean()
            sems = n.loc[n['condition']==i].groupby(['pupil_b_bin']).sem()
            plt.errorbar(x=means['pupil_b'], y=means[m], xerr=sems['pupil_b'], yerr=sems[m], fmt='o', color=sns.color_palette(color_blend,3)[i-1], alpha=0.7)

        plt.ylabel('Δ Pupil response (% change)')
        coefs = ['SOA:           ', 'Pupil:          ', 'Interaction: ']
        for i in range(3):
            coef = coefs[i]
            f = res.anova_table["F Value"][i] #FIXME so far this is a 2-factor anova, here we're accessing the condition coefficient which is not complete and not the one of interest
            df1 = res.anova_table["Num DF"][i]
            df2 = res.anova_table["Den DF"][i]
            s = res.anova_table["Pr > F"][i]
            if s < 0.0001:
                txt = ' p < 0.001'
            else:
                txt = ' p = {}'.format(round(s,3))
            if s<0.05:
                plt.text(x=-21, y=y-(i*2.3), s='{}'.format(coef)+'$F_{{({},{})}}=$'.format(df1, df2)+str(round(f,3))+','+r'$\bf{{{}}}$'.format(txt), size=7, va='center', ha='left') # , c=color_AS)
            else:
                plt.text(x=-21, y=y-(i*2.3), s='{}'.format(coef)+'$F_{{({},{})}}=$'.format(df1, df2)+str(round(f,3))+','+txt, size=7, va='center', ha='left') # , c=color_AS)
    
    if m == 'int_abs_diff':
        ## model for condition, baseline and interaction effects in psychometric function
        model1 = AnovaRM(data=pf1, depvar=m, subject='subject_id', within=['condition', 'pupil_b_bin'])
        res1 = model1.fit()
        # ## models for baseline effects in differences between each condition and no-as condition in absolute shift
        # model1 = AnovaRM(data=pf0, depvar='int_abs_diff_1', subject='subject_id', within=['pupil_b_bin'])
        # model2 = AnovaRM(data=pf0, depvar='int_abs_diff_2', subject='subject_id', within=['pupil_b_bin'])
        # model3 = AnovaRM(data=pf0, depvar='int_abs_diff_3', subject='subject_id', within=['pupil_b_bin'])

        # res1 = model1.fit()
        # res2 = model2.fit()
        # res3 = model3.fit()


        ##plot for condition, baseline and interaction effects in psychometric function
        # sns.pointplot(data=pf, x='pupil_b_bin', y=m, hue='condition', palette = [color_noAS, sns.color_palette(color_blend,3)[0],sns.color_palette(color_blend,3)[1],sns.color_palette(color_blend,3)[2]]) 
        # sns.lineplot(data=pf1, x='pupil_b_bin', y=m, hue='condition', errorbar='se', palette = [sns.color_palette(color_blend,3)[0],sns.color_palette(color_blend,3)[1],sns.color_palette(color_blend,3)[2]]) 
        for i in [1,2,3]:
            means = df_res.loc[df_res['condition']==i].groupby(['pupil_b_bin']).mean()
            sems = df_res.loc[df_res['condition']==i].groupby(['pupil_b_bin']).sem()
            means1 = n.loc[n['condition']==i].groupby(['pupil_b_bin']).mean()
            sems1 = n.loc[n['condition']==i].groupby(['pupil_b_bin']).sem()
            means1['pupil_b'] = means['pupil_b']
            sems1['pupil_b'] = sems['pupil_b']
            plt.errorbar(x=means1['pupil_b'], y=means1[m], xerr=sems1['pupil_b'], yerr=sems1[m], fmt='o', color=sns.color_palette(color_blend,3)[i-1], alpha=0.7)


        ## plots for baseline effects in differences between each condition and no-as condition in absolute shift
        # sns.pointplot(data=pf0, x='pupil_b_bin', y='int_abs_diff_1', color = sns.color_palette(color_blend,3)[0]) 
        # sns.pointplot(data=pf0, x='pupil_b_bin', y='int_abs_diff_2', color = sns.color_palette(color_blend,3)[1]) 
        # sns.pointplot(data=pf0, x='pupil_b_bin', y='int_abs_diff_3', color = sns.color_palette(color_blend,3)[2]) 
        # plt.setp(ax.collections, alpha=.3) #for the markers
        # plt.setp(ax.lines, alpha=.3)       #for the lines

        # sns.lineplot(data=pf0, x='pupil_b_bin', y='int_abs_diff_1', errorbar='se', color = sns.color_palette(color_blend,3)[0]) 
        # sns.lineplot(data=pf0, x='pupil_b_bin', y='int_abs_diff_2', errorbar='se', color = sns.color_palette(color_blend,3)[1]) 
        # sns.lineplot(data=pf0, x='pupil_b_bin', y='int_abs_diff_3', errorbar='se', color = sns.color_palette(color_blend,3)[2]) 
        # for i,j in zip([0,1,2],[res1,res2,res3]):
        #     f = j.anova_table["F Value"][0] 
        #     df1 = j.anova_table["Num DF"][0]
        #     df2 = j.anova_table["Den DF"][0]
        #     s = j.anova_table["Pr > F"][0]
        coefs = ['SOA:           ', 'Pupil:          ', 'Interaction: ']
        for i in range(3):
            coef = coefs[i]
            f = res1.anova_table["F Value"][i] #FIXME so far this is a 2-factor anova, here we're accessing the condition coefficient which is not complete and not the one of interest
            df1 = res1.anova_table["Num DF"][i]
            df2 = res1.anova_table["Den DF"][i]
            s = res1.anova_table["Pr > F"][i]
            if s < 0.0001:
                txt = ', p < 0.001'
            else:
                txt = ', p = {}'.format(round(s,3))
            plt.text(x=-21, y=y-0.04*i, s= '{}'.format(coef)+'$F_{{({},{})}}=$'.format(df1, df2)+str(round(f,3))+txt, size=7, va='center', ha='left') # , c=sns.color_palette(color_blend,3)[i])
        plt.ylabel('Δ | Shift |')

    plt.ylim(z[0],z[1])
    plt.xlabel('Baseline pupil size (% w.r.t. mean)')
    # plt.xticks([0,1,2])
    plt.legend([], [], frameon=False)
    sns.despine(trim=True)
    plt.tight_layout()
    # fig.savefig(os.path.join(figs_dir, '{}_{}_on_baseline_bin.pdf'.format(data_set, m)), bbox_inches='tight')



# ------------------------------------------------------------------------
# Fig S1: means of behavior 
# ------------------------------------------------------------------------
df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id'])
pf = df.loc[df['trial_nr']>trial_cutoff,:].groupby(['subject_id']).apply(compute_kernel2).reset_index()
pf['intercept_abs'] = abs(pf['Intercept'])
pf['dv'] = (pf['dv0_n'] +pf['dv1_n'] + pf['dv2_n'] + pf['dv3_n'] + pf['dv4_n'] + pf['dv5_n'] + pf['dv6_n'] + pf['dv7_n'])/8
pf['dv_abs'] = abs(pf['dv'])

def plot_means(df_res, measure):
    # df_res = utils.compute_results(df=df.loc[:,:], groupby=['subject_id'])
    # means = df_res[measure].mean()
    # sems = df_res[measure].sem()
    print('mean {}'.format(measure), df_res[measure].mean())
    print('sem {}'.format(measure), df_res[measure].sem())

    fig = plt.figure(figsize=(1.3,2))
    ax = fig.add_subplot() 

    sns.stripplot(x=0, y=df_res[measure], color='grey', ax=ax, linewidth=0.2, alpha=0.7)
    ax.hlines(y=df_res[measure].mean(), xmin=-0.4, xmax=0.4, zorder=10, colors='k')
    ax.set_xlim([-1,1])
    ax.get_xaxis().set_visible(False)

    if measure == 'pupil_r_diff':
        plt.ylabel('Pupil response (% change)')
    if measure == 'intercept_abs':
        ax.set_ylabel('| Shift |')
    if measure == 'Intercept':
        ax.set_ylabel('Shift')
    if measure == 'dv':
        ax.set_ylabel("Slope")
    if measure == 'rt':
        plt.ylabel('Reaction time')
    if measure == 'correct':
        plt.ylabel('Accuracy')

    sns.despine(trim=True)
    plt.tight_layout()

    return fig

for m in ['intercept_abs','Intercept','dv']:
    fig = plot_means(df_res=pf, measure=m)
    # fig.savefig(os.path.join(figs_dir, '{}_{}_mean.pdf'.format(data_set, m)))

for m in ['rt']:
    fig = plot_means(df_res, measure=m)
    # fig.savefig(os.path.join(figs_dir, '{}_{}_mean.pdf'.format(data_set, m)))

# ----------------------------------------------------------------------
# Fig S2: pupil size metrics (baseline, TPR, task-irrelevant sound evoked pupil response) across trials (normal trials)
# -----------------------------------------------------------------------
## tpr and bl across trials by condition:
n_jobs = 4
n_boot = 500
means = []
sems = []
for c in [0,1,2,3]:
    means.append(compute_results(df.loc[df['condition']==c,:], groupby=['trial_nr']))
    sem_ = pd.concat(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(compute_results)
                                                (df.loc[df['condition']==c,:].sample(frac=1, replace=True), ['trial_nr'], i)
                                                for i in range(n_boot)))
    sems.append((sem_.groupby(['trial_nr']).quantile(0.84) - sem_.groupby(['trial_nr']).quantile(0.16))/2)
for y in ['pupil_b', 'tpr', 'pupil_r_diff']:
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot()
    plt.axvline(trial_cutoff, color='k', lw=0.5)
    color_choices = [color_noAS,sns.color_palette(color_blend,3)[0],sns.color_palette(color_blend,3)[1],sns.color_palette(color_blend,3)[2]]
    for i in range(len(means)):
        x = np.array(means[i]['trial_nr'])
        window = 10
        mean = means[i].rolling(window=window, center=True, min_periods=1).mean()
        sem = sems[i].rolling(window=window, center=True, min_periods=1).mean()
        sem = sem.reset_index()
        if y == 'pupil_r_diff':
            if int(mean['condition'].mean()) == 1:
                plt.fill_between(x, mean[y]-sem[y], mean[y]+sem[y], alpha=0.2, color=color_choices[int(mean['condition'].mean())])
                plt.scatter(x, mean[y], color=color_choices[int(mean['condition'].mean())], s=2.7, marker='o', lw=0.05, alpha=0.7)
        elif (y == 'rt') | (y == 'correct') | (y == 'd') | (y == 'c') | (y == 'c_abs') | (y == 'pupil_b') | (y == 'tpr'):
            plt.fill_between(x, mean[y]-sem[y], mean[y]+sem[y], alpha=0.2, color=color_choices[int(mean['condition'].mean())])
            plt.scatter(x, mean[y], color=color_choices[int(mean['condition'].mean())], s=2.7, marker='o', lw=0.05, alpha=0.7)
    plt.xlabel('Trial (#)')
    if y == 'pupil_r_diff':
        # ax.set_title('Pupil response to AS', fontweight='bold')
        plt.ylabel('Pupil response (% change)')
    if y == 'tpr':
        # ax.set_title('TPR', fontweight='bold')
        plt.ylabel('Pupil response (% change)')
    if y == 'pupil_b':
        # ax.set_title('Pupil baseline', fontweight='bold')
        plt.ylabel('Relative pupil size (%)')
    if y == 'c_abs':
        # ax.set_title('Bias', fontweight='bold')
        ax.set_ylabel('| Criterion c |')
    if y == 'c':
        # ax.set_title('Bias',fontweight='bold')
        ax.set_ylabel('Criterion c')
    if y == 'd':
        # ax.set_title('Performance',fontweight='bold')
        ax.set_ylabel("Sensitivity d'")
    if y == 'rt':
        # ax.set_title('Reaction time',fontweight='bold')
        plt.ylabel('Reaction time')
    if y == 'correct':
        plt.ylabel('Accuracy')
    plt.xlabel('Trial (#)')    
    sns.despine(trim=True)
    plt.tight_layout()
    # fig.savefig(os.path.join(figs_dir, '{}_across_trials_{}.pdf'.format(data_set, y)))

# ------------------------------------------------------------------
# Fig. S6 Variability of task-irrelevant stimulus-evoked pupil responses
# ------------------------------------------------------------------
# correlation size of task-irrelevant evoked pupil response (by bin) and behaviour 
df['AS'] = 0
df.loc[(df['condition']!=0),'AS'] = 1
groupby=['subject_id', 'AS', 'pupil_r_diff_bin']
indices=['int_abs_diff']
r_place = 5
df_res = utils.compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=groupby)
df_res['c_abs_diff'] = np.NaN
for s in df_res['subject_id'].unique():
    df_res.loc[(df_res['subject_id']==s),'c_abs_diff'] = df_res.loc[(df_res['subject_id']==s),'c_abs'] - float(df_res.loc[ (df_res['subject_id']==s) & (df_res['condition']==0), 'c_abs'].values)
df_res = df_res.loc[df_res['AS']==1,:]

pf = df.loc[df['trial_nr']>trial_cutoff,:].groupby(groupby).apply(compute_kernel2).reset_index() #FIXME run again now with cut-off
pf['int_abs'] = abs(pf['Intercept'])
pf['dv'] = (pf['dv0_n'] +pf['dv1_n'] + pf['dv2_n'] + pf['dv3_n'] + pf['dv4_n'] + pf['dv5_n'] + pf['dv6_n'] + pf['dv7_n'])/8
pf['dv_abs'] = abs(pf['dv'])
pf['int_abs_diff'] = np.NaN
for s in pf['subject_id'].unique():
    pf.loc[(pf['subject_id']==s),'int_abs_diff'] = pf.loc[(pf['subject_id']==s),'int_abs'] - float(pf.loc[(pf['subject_id']==s) & (pf['AS']==0), 'int_abs'].values)
pf = pf.loc[pf['AS']==1,:]
pf['pupil_r_diff'] = df_res['pupil_r_diff']

fig = plt.figure(figsize=((2*len(indices)),2))
plt_nr = 1
for m in indices:
    ax = fig.add_subplot(1,len(indices),plt_nr)
    means = pf.groupby(['pupil_r_diff_bin']).mean()
    sems = pf.groupby(['pupil_r_diff_bin']).sem()

    model = smf.mixedlm("int_abs_diff ~ pupil_r_diff", groups=pf["subject_id"], re_formula="~pupil_r_diff", data=pf) # re_formula="0", #TODO check if correct
    result = model.fit()

    r = []
    for subj, d in pf.groupby(['subject_id']):
        r.append(sp.stats.pearsonr(d['pupil_r_diff'],d[m])[0])
    t, p = sp.stats.ttest_1samp(r, 0)
    plt.errorbar(x=means['pupil_r_diff'], y=means[m], yerr=sems[m], fmt='o', color=color_AS, alpha=0.7)
    if p < 0.05:
        sns.regplot(x=means['pupil_r_diff'], y=means[m], scatter=False, ci=None, color='k')
    s = round(p,3)
    if s == 0.0:
        txt = ', p < 0.001'
    else:
        txt = ', p = {}'.format(s)
    if m=='rt':
        plt.text(x=r_place, y=max(means[m])+0.04, s='r='+str(round(statistics.mean(r),3))+txt, size=7, va='center', ha='center')
    else:        
        plt.text(x=r_place, y=max(means[m])+0.08, s='r='+str(round(statistics.mean(r),3))+txt, size=7, va='center', ha='center')
    plt.xlabel('Δ Pupil response\n(% signal change)')
    if m == 'int_abs_diff':
        plt.ylabel('Δ | Shift |')
sns.despine(trim=True)
plt.tight_layout()
# fig.savefig(os.path.join(figs_dir, '{}_as_size_behavior_{}.pdf'.format(data_set, m)))

###################### OLD #########################################

# ----------------------------------------------------------------------------------------
# Fig 2: tpr w/ behaviour correlation analysis: #TODO make plotting function and move to a utils script
# ----------------------------------------------------------------------------------------
## using sdt
fig = plt.figure(figsize=(6,2))
plt_nr = 1
df_res = compute_results(df=df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition', 'tpr_bin'])
df_res = df_res.loc[df_res['condition']==0,:].reset_index()
for m in ['d', 'c', 'c_abs']:
    ax = fig.add_subplot(1,3,plt_nr)
    means = df_res.groupby(['tpr_bin']).mean()
    sems = df_res.groupby(['tpr_bin']).sem()

    r = []
    for subj, d in df_res.groupby(['subject_id']):
        r.append(sp.stats.pearsonr(d['tpr_c'],d[m])[0])
    t, p = sp.stats.ttest_1samp(r, 0)
    plt.errorbar(x=means['tpr_c'], y=means[m], yerr=sems[m], fmt='o', color=color_noAS)
    plt.title('p = {}'.format(round(p,3)))
    if p < 0.05:
        sns.regplot(x=means['tpr_c'], y=means[m], scatter=False, ci=None, color='k')
    plt.title('p = {}'.format(round(p,3)))
    plt.xlabel('Pupil response\n(% signal change)')
    plt.ylabel(m)
    plt_nr+=1
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figs_dir, '{}_tpr_c_abs.pdf'.format(data_set)))

## using psychometric funtion (slope and shift) where shift would indicate bias

res = df.loc[df['condition']==0].groupby(['subject_id', 'dv_bin', 'tpr_bin']).mean().reset_index()
sns.set_palette("blend:#DECA59,#E06138",9)
fig = plt.figure(figsize=(2.5,2))
ax = fig.add_subplot(111)
sns.pointplot(x='dv_bin', y='choice', hue='tpr_bin', errorbar='se', data=res, hue_order=[0,1,2,3,4,5,6,7], scale=0.4, ax=ax)
plt.xlabel('DV bin')
plt.ylabel('% Diagonal responses')
ax.legend(title='TPR bin', title_fontsize=6, loc='center left', bbox_to_anchor=(1, 0.5))
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figs_dir, '{}_psychometric_function.pdf'.format(data_set)))

pf = df.loc[df['condition']==0].groupby(['subject_id','tpr_bin']).apply(compute_psychometric_function).reset_index()
pf['intercept_abs'] = abs(pf['Intercept'])
pf['dv_abs'] = abs(pf['dv'])
pf['tpr_c'] = df_res['tpr_c']
for m in ['dv', 'Intercept', 'intercept_abs']:
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(1,1,1)
    #sns.pointplot(x='tpr_bin', y=m, errorbar='se', data=pf, ax=ax)
    means = pf.groupby(['tpr_bin']).mean().reset_index()
    sems = pf.groupby(['tpr_bin']).sem().reset_index()
    r = []
    for subj, d in pf.groupby(['subject_id']):
        r.append(sp.stats.pearsonr(d['tpr_c'],d[m])[0])
    t, p = sp.stats.ttest_1samp(r, 0)
    plt.errorbar(x=means['tpr_c'], y=means[m], yerr=sems[m], fmt='o', color=color_noAS)
    plt.title('p = {}'.format(round(p,3)))
    if p < 0.05:
        sns.regplot(x=means['tpr_c'], y=means[m], scatter=False, ci=None, color='k', ax=ax)
    plt.xlabel('TPR\n(% signal change)') #use \n to start new line
    if m == 'intercept_abs':
        plt.ylabel('Absolute shift')
    if m == 'dv':
        plt.ylabel('Slope')
    if m == 'Intercept':
        plt.ylabel('Shift')
    sns.despine(trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(figs_dir, '{}_tpr_{}.pdf'.format(data_set,m)))

## normal pupil time course 
epochs_p_resp = epochs_p_resp.reset_index()
epochs_p_resp['tpr_bin'] = df['tpr_bin']
columns = ['subject_id', 'block_id', 'trial_nr', 'condition', 'tpr_bin']
epochs_p_resp = epochs_p_resp.set_index(columns)

x = epochs_p_resp.columns
fig = utils_amsterdam_orientation_discrimination.plot_normal_pupil(epochs_p_resp.query('trial_nr > {}'.format(trial_cutoff)).loc[:, (x>-3)&(x<2)], locking='resp', draw_box=True, shade=False)
fig.savefig(os.path.join(figs_dir, '{}_normal_pupil_resp.pdf'.format(data_set)))

x = epochs_p_stim.columns
fig = utils_amsterdam_orientation_discrimination.plot_normal_pupil(epochs_p_stim.query('trial_nr > {}'.format(trial_cutoff)).loc[:, (x>-1.5)&(x<4)], shade=True)
fig.savefig(os.path.join(figs_dir, '{}_normal_pupil_bl_shade.pdf'.format(data_set)))

## for JW normal stim locked time course by correct
x = epochs_p_stim.columns
epochs_p_stim = epochs_p_stim.reset_index()
epochs_p_stim['correct'] = df['correct']
columns = ['subject_id', 'block_id', 'trial_nr', 'condition', 'correct']
epochs_p_stim = epochs_p_stim.set_index(columns)
epochs = epochs_p_stim.query('trial_nr > {}'.format(trial_cutoff)).loc[:, (x>-1.5)&(x<4)]
means = epochs.groupby(['subject_id', 'condition', 'correct']).mean().groupby(['condition', 'correct']).mean()
sems = epochs.groupby(['subject_id', 'condition', 'correct']).mean().groupby(['condition', 'correct']).sem()
means_c = means.loc[(means.index.get_level_values(1)==1) & (means.index.get_level_values(0)=='normal')]
means_e = means.loc[(means.index.get_level_values(1)==0) & (means.index.get_level_values(0)=='normal')]
sems_c = sems.loc[(sems.index.get_level_values(1)==1) & (sems.index.get_level_values(0)=='normal')]
sems_e = sems.loc[(sems.index.get_level_values(1)==0) & (sems.index.get_level_values(0)=='normal')]

fig = plt.figure(figsize=(2,2))
ax = fig.add_subplot(111)
x = np.array(means_c.columns, dtype=float)
plt.axvline(0, color='k', ls='--', linewidth=0.5)

plt.fill_between(x, means_c.iloc[0]-sems_c.iloc[0], 
                        means_c.iloc[0]+sems_c.iloc[0], alpha=0.25, color=color_AS)
plt.plot(x, means_c.iloc[0], color=color_AS)

plt.fill_between(x, means_e.iloc[0]-sems_e.iloc[0], 
                        means_e.iloc[0]+sems_e.iloc[0], alpha=0.25, color=color_noAS)
plt.plot(x, means_e.iloc[0], color=color_noAS)

# plt.xticks([-4,-3,-2,-1,0,1,2])
# plt.yticks([-2,-1,0,1])
plt.xlabel('Time from 1st stimulus (s)')


plt.ylabel('Pupil response (% change)')
# plt.legend()

sns.despine(trim=True)
plt.tight_layout()
# p_thresh = 0.5
# pf['x_tpr_bin'] = (math.log(p_thresh/(1-p_thresh))-pf['Intercept'])/pf['dv']

# ## plot sigmoid function with params from logistic regression
# coef=pf['dv'].mean()
# intercept = pf['Intercept'].mean()
# def sigmoid(x):
#     z = np.dot(coef, x) + intercept
#     return 1 / (1 + np.exp(-z))

# # Generate input values within the desired range
# start = -1  # define the start point
# end = 1  # define the end point
# num_points = 11  # define the number of points
# x = np.linspace(start, end, num_points)

# # Calculate predicted probabilities using the sigmoid function
# y = sigmoid(x)

# # Plot the sigmoid function
# plt.plot(x, y)

# -----------------------------------------------------------------------------------
# Fig: plot diff pupil responses:
# -----------------------------------------------------------------------------------
x = epochs_p_stim.columns
fig = utils_amsterdam_orientation_discrimination.plot_pupil_responses_amsterdam_orientation(epochs_p_stim.query('trial_nr > {}'.format(trial_cutoff)).loc[:, (x>-1)&(x<4)].diff(axis=1))
fig.savefig(os.path.join(figs_dir, '{}_pupil_responses_diff.pdf'.format(data_set)))

# -----------------------------------------------------------------------------------
# Fig: behavioral analyses signal detection theory
# -----------------------------------------------------------------------------------
trial_cutoff = 19
df_res = compute_results(df.loc[df['trial_nr']>trial_cutoff,:], groupby=['subject_id', 'condition'])
for m in ['d', 'c', 'c_abs', 'rt']: # , 'pupil_r_diff']:  ['rt', 'correct',
    if m == 'rt':
        fig = plt.figure(figsize=(2,2.5))
    else:
        fig = plt.figure(figsize=(2,2)) #2.5))
    ax = fig.add_subplot() #121)
    # sns.stripplot(x=0, y=df_res.loc[df_res['condition']==0, m], color=color_noAS, linewidth=0.2)
    # ax = fig.add_subplot(122)
    y = df_res.loc[df_res['condition']==1, m].values-df_res.loc[df_res['condition']==0, m].values
    sns.stripplot(x=1, y=y, 
                color=sns.color_palette(color_blend,3)[0], ax=ax, linewidth=0.2, alpha=0.7, edgecolor=sns.color_palette(color_blend,3)[0])
    ax.hlines(y=y.mean(), xmin=(0)-0.4, xmax=(0)+0.4, zorder=10, colors='k')
    y = df_res.loc[df_res['condition']==2, m].values-df_res.loc[df_res['condition']==0, m].values
    sns.stripplot(x=2, y=y, 
                color=sns.color_palette(color_blend,3)[1], ax=ax, linewidth=0.2, alpha=0.7, edgecolor=sns.color_palette(color_blend,3)[1])
    ax.hlines(y=y.mean(), xmin=(1)-0.4, xmax=(1)+0.4, zorder=10, colors='k')
    df_res.loc[df_res['condition']==3, m].values-df_res.loc[df_res['condition']==0, m].values
    sns.stripplot(x=3, y=y, 
                color=sns.color_palette(color_blend,3)[2], ax=ax, linewidth=0.2, alpha=0.7, edgecolor=sns.color_palette(color_blend,3)[2])
    ax.hlines(y=y.mean(), xmin=(2)-0.4, xmax=(2)+0.4, zorder=10, colors='k')
    p_values = []
    bf_values = []
    for c in [1,2,3]:
        p_values.append(sp.stats.ttest_rel(df_res.loc[df_res['condition']==c, m].values,
                        df_res.loc[df_res['condition']==0, m].values)[1])
        t_value = sp.stats.ttest_rel(df_res.loc[df_res['condition']==c, m].values,
                        df_res.loc[df_res['condition']==0, m].values)[0]
        bf_values.append(bayesfactor_ttest(t_value, len(df['subject_id'].unique()), paired=True))
        # bf_values.append(ttest(df_res.loc[df_res['condition']==c, m].values, df_res.loc[df_res['condition']==0, m].values, paired=True)['BF10'])
    p_values = [p * 3 for p in p_values]
    model_anova = AnovaRM(data=df_res, depvar=m, subject='subject_id', within=['condition'])
    print(model_anova.fit())
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    if m == 'rt':
        for j, cond in enumerate([1,2,3]):
            # plt.text(x=j, y=1, s='$BF_{{{}}}={}$'.format('10',round(bf_values[j],3)), size=5, rotation=90, transform=trans)
            if m == 'rt':
                plt.text(x=j, y=0.9, s='$BF_{{{0:}}}={1: .3g}$'.format('10',round(bf_values[j],3)), size=5, rotation=90, transform=trans)
            else:
                plt.text(x=j, y=1, s='$BF_{{{0:}}}={1: .3g}$'.format('10',round(bf_values[j],3)), size=5, rotation=90, transform=trans)
    plt.axhline(0, color='black', lw=0.5)
    if m == 'pupil_r_diff':
        plt.ylabel('Δ Pupil response (% change)')
    if m == 'c_abs':
        # ax.set_title('Bias AS - no AS', pad=35,fontweight='bold')
        plt.ylabel('Δ | Criterion c |')
    if m == 'c':
        # ax.set_title('Bias AS - no AS', pad=35,fontweight='bold')
        plt.ylabel('Δ Criterion c')
    if m == 'd':
        # ax.set_title('Performance AS - no AS', pad=35,fontweight='bold')
        plt.ylabel("Δ Sensitivity d'")
    if m == 'rt':
        plt.ylim(-0.08, 0.09)
        # ax.set_title('Reaction time AS - no AS', pad=35,fontweight='bold')
        plt.ylabel("Δ Reaction time")
    # ax.set_xticklabels(rotation=90)
    plt.xlabel('Time from 1st stimulus (s)')
    ax.set_xticklabels([-1, 0, 1])
    sns.despine(trim=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    fig.savefig(os.path.join(figs_dir, '{}_as_behavior_{}.pdf'.format(data_set, m)))

shell()

# plot psychophysical kernel:
fig = plot_psycho_kernel_logit(df.loc[df['trial_nr']>trial_cutoff,:])
fig.savefig(os.path.join(figs_dir, '{}_psycho_kernel.pdf'.format(data_set)))

fig = plot_psycho_regression(df.loc[df['trial_nr']>trial_cutoff,:])
fig.savefig(os.path.join(figs_dir, '{}_psycho_kernel_regression.pdf'.format(data_set)))

## plot bias measures:
fig = plot_bias_measures(df.loc[df['trial_nr']>trial_cutoff,:])
fig.savefig(os.path.join(figs_dir, '{}_bias_measures.pdf'.format(data_set)))





res = df.groupby(['subject_id', 'dv_bin', 'condition']).mean().reset_index()
fig = plt.figure(figsize=(2,2))
ax = fig.add_subplot(111)
sns.pointplot(x='dv_bin', y='choice', hue='condition', errorbar='se', data=res, hue_order=['normal', 'AS_0', 'AS_1', 'AS_2'], ax=ax)
plt.xlabel('DV bin')
plt.ylabel('% Diagonal responses')
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figs_dir, 'psychometric_function.pdf'))





shell()


res = df.groupby(['subject_id', 'condition', 'dv_abs_bin'])[['correct', 'rt', 'choice']].mean().reset_index()
sns.pointplot(x='dv_abs_bin', y='choice', hue='condition', data=res, errorbar='se')

aovrm = AnovaRM(res.loc[(res['condition']=='normal')|(res['condition']=='AS_1')], 'choice', 'subject_id', within=['condition', 'dv_abs_bin'], aggregate_func='mean').fit()
print(aovrm.summary())







def permutationTest_correlation(x1, x2, nrand=1000):
    rng = np.random.default_rng()
    data = np.vstack((x1,x2)).T
    corrrand = np.zeros(nrand)
    for i in range(nrand):
        data = rng.permuted(data, axis=1)
        corrrand[i] = sp.stats.pearsonr(data[:,0], data[:,1]-data[:,0])[0]
    truecorr = sp.stats.pearsonr(x1, x2-x1)[0]
    p_value = (corrrand<truecorr).mean()
    # p_value = (abs(corrrand) >= abs(truecorr)).mean()
    return truecorr, corrrand, p_value


fig = plt.figure(figsize=(6,6))
plt_nr = 1
for c in ['AS_0', 'AS_1', 'AS_2',]:
    for m in ['Intercept', 'stimulus_p', 'choice_p']:
        ax = fig.add_subplot(3,3,plt_nr)
        plt.scatter(coefs.loc[coefs['condition']=='normal', m],
                    coefs.loc[coefs['condition']==c, m].values-coefs.loc[coefs['condition']=='normal', m].values)
        
        truecorr, corrrand, p_value = permutationTest_correlation(coefs.loc[coefs['condition']=='normal', m].values,
                                                                  coefs.loc[coefs['condition']==c, m].values)
        
        plt.title('r={} p={}'.format(round(truecorr,2), round(p_value,3)))
        plt_nr += 1

# sns.despine(trim=True)
# plt.tight_layout()






#     plt.figure()
#     sns.pointplot(x='condition', y=c, data=psycho, errorbar='se')




# df.loc[:, 'pupil_b_median_split'] = df.loc[:,:].groupby(['subject_id', 'block_id', 'condition'])['pupil_b'].apply(pd.qcut, q=bins, labels=False)


# shell()

# for b in range(3):
#     fig = plot_psycho_kernel_logit(df.loc[df['dv_abs_bin']==b,:])
#     fig.savefig(os.path.join(figs_dir, 'psycho_kernel_{}.pdf'.format(b)))
# for b in range(5):
#     fig = plot_psycho_kernel_logit(df.loc[df['pupil_b_median_split']==b,:])
#     fig.savefig(os.path.join(figs_dir, 'psycho_kernel_pupil_b_{}.pdf'.format(b)))

# df_res, _, _ = compute_results(df, groupby=['subject_id', 'condition'])

# fig = plot_psycho_kernel2(df, normalize=True)
# fig.savefig(os.path.join(figs_dir, 'psycho_kernel_individuals.pdf'))



fig = plt.figure(figsize=(2,2))
ax = fig.add_subplot(111)
sns.pointplot(x='dv_bin', y='tpr', hue='condition', errorbar='se', data=res, hue_order=['normal', 'AS_0', 'AS_1', 'AS_2'], ax=ax)
plt.xlabel('DV bin')
plt.ylabel('Pupil response')
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figs_dir, 'psychometric_function_tpr.pdf'))

bins = 5
df.loc[:, 'pupil_b_median_split'] = df.loc[:,:].groupby(['subject_id', 'block_id', 'condition'])['pupil_b'].apply(pd.qcut, q=bins, labels=False)
bin_by = 'pupil_b_median_split'
df_res = df.groupby(['subject_id', 'condition', 'pupil_b_median_split']).apply(utils.sdt, stim_column='stimulus', response_column='response').reset_index()
df_res['c_abs'] = abs(df_res['c'])
fig = plt.figure(figsize=(6,6))
plt_nr = 1
for c in ['AS_0', 'AS_1', 'AS_2']:
    for i, y in enumerate(['d', 'c', 'c_abs']):
        ax = fig.add_subplot(3,3,plt_nr)
        sns.pointplot(x=bin_by, y=y, hue='condition', hue_order=['normal', c], data=df_res, errorbar='se')
        aovrm = AnovaRM(df_res.loc[(df_res['condition']=='normal')|(df_res['condition']==c)], y, 'subject_id', within=['condition', bin_by], aggregate_func='mean').fit()
        print(aovrm.summary())
        plt.title('p = {}'.format(round(aovrm.anova_table['Pr > F'][2],3)))
        ax.get_legend().remove()
        plt_nr += 1
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figs_dir, 'baseline_pupil_interaction.pdf'))

bins = 5
df.loc[:, 'dv_abs_median_split'] = df.loc[:,:].groupby(['subject_id', 'block_id', 'condition'])['dv_abs'].apply(pd.qcut, q=bins, labels=False)
bin_by = 'dv_abs_median_split'
df_res = df.groupby(['subject_id', 'condition', 'dv_abs_median_split']).apply(utils.sdt, stim_column='stimulus', response_column='response').reset_index()
df_res['c_abs'] = abs(df_res['c'])
fig = plt.figure(figsize=(6,6))
plt_nr = 1
for c in ['AS_0', 'AS_1', 'AS_2']:
    for i, y in enumerate(['d', 'c', 'c_abs']):
        ax = fig.add_subplot(3,3,plt_nr)
        sns.pointplot(x=bin_by, y=y, hue='condition', hue_order=['normal', c], data=df_res, errorbar='se')
        aovrm = AnovaRM(df_res.loc[(df_res['condition']=='normal')|(df_res['condition']==c)], y, 'subject_id', within=['condition', bin_by], aggregate_func='mean').fit()
        print(aovrm.summary())
        plt.title('p = {}'.format(round(aovrm.anova_table['Pr > F'][2],3)))
        ax.get_legend().remove()
        plt_nr += 1
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figs_dir, 'dv_abs_interaction.pdf'))

shell()


pf = df.groupby(['subject_id', 'condition']).apply(compute_psychometric_function).reset_index()
sns.pointplot(x='condition', y='Intercept', errorbar='se', data=pf)
sns.pointplot(x='condition', y='dv', errorbar='se', data=pf)


psycho = df.groupby(['subject_id', 'condition']).apply(compute_kernel)

i_n = psycho.loc[psycho.index.get_level_values('condition')=='normal', 'Intercept'].values
i_0 = psycho.loc[psycho.index.get_level_values('condition')=='AS_0',   'Intercept'].values
i_1 = psycho.loc[psycho.index.get_level_values('condition')=='AS_1',   'Intercept'].values
i_2 = psycho.loc[psycho.index.get_level_values('condition')=='AS_2',   'Intercept'].values

res = df.groupby(['subject_id', 'condition'])['tpr'].mean()
p_n = res.loc[psycho.index.get_level_values('condition')=='normal',:].values
p_0 = res.loc[psycho.index.get_level_values('condition')=='AS_0',:].values
p_1 = res.loc[psycho.index.get_level_values('condition')=='AS_1',:].values
p_2 = res.loc[psycho.index.get_level_values('condition')=='AS_2',:].values

print(sp.stats.pearsonr(i_0-i_n, i_n))
print(sp.stats.pearsonr(i_1-i_n, i_n))
print(sp.stats.pearsonr(i_2-i_n, i_n))


print(sp.stats.pearsonr(i_0-i_n, p_0-p_n))
print(sp.stats.pearsonr(i_1-i_n, p_1-p_n))
print(sp.stats.pearsonr(i_2-i_n, p_2-p_n))




#TODO: 
# - interaction between choice history as AS!


df.groupby(['stimulus'])

df.loc[df['stimulus']==0, 'dv3']

for i in range(7):

    plt.figure()
    ind1 = np.array((df['condition']=='normal')&(df['stimulus']==0)&(df['response']==0)&(df['dv{}'.format(i)]>0))
    ind2 = np.array((df['condition']=='normal')&(df['stimulus']==1)&(df['response']==1)&(df['dv{}'.format(i)]<0))
    ind3 = np.array((df['condition']=='normal')&(df['stimulus']==0)&(df['response']==0)&(df['dv{}'.format(i)]<0))
    ind4 = np.array((df['condition']=='normal')&(df['stimulus']==1)&(df['response']==1)&(df['dv{}'.format(i)]>0))
    
    x = epochs_p_stim.columns
    plt.plot(epochs_p_stim.loc[(ind1|ind2),(x>-1)&(x<4)].mean(axis=0), label='correct')
    plt.plot(epochs_p_stim.loc[(ind3|ind4),(x>-1)&(x<4)].mean(axis=0), label='error')
    plt.legend()
    




plt.figure()
ind1 = np.array((df['condition']=='normal')&(df['correct']==1))
ind2 = np.array((df['condition']=='normal')&(df['correct']==0))
x = epochs_p_stim.columns
plt.plot(epochs_p_stim.loc[ind1,(x>-1)&(x<4)].mean(axis=0))
plt.plot(epochs_p_stim.loc[ind2,(x>-1)&(x<4)].mean(axis=0))

shell()


# res = (df.loc[df['stimulus']=='cardinal',:].groupby(['subject_id', 'condition'])['correct'].mean()-
#         df.loc[df['stimulus']=='diagonal',:].groupby(['subject_id', 'condition'])['correct'].mean())

# df['diff_bin'] = df.groupby(['subject_id', 'block_id'])['difficulty_actual'].apply(pd.qcut, q=5, labels=False)









# a = df.groupby(['subject_id', 'block_id']).count()
# print(a.loc[a['trial_nr']>100,:])

# # # sort:
# # sort = np.argsort(df['subject_id', ''])
# # df = df.iloc[sort]
# # epochs_p_stim = epochs_p_stim.iloc[sort]
# # epochs_p_resp = epochs_p_resp.iloc[sort]
# # epochs_b_resp = epochs_b_resp.iloc[sort]

# # order:
# df = df.groupby(['subject_id', 'block_id']).apply(utils.add_condition_order).reset_index(drop=True)

# # sequential:
# df = df.groupby(['subject_id', 'block_id']).apply(utils.compute_sequential).reset_index(drop=True)

# # map responses:
# ind = (df['subject_id']<=3)|(df['subject_id']>=25)
# df.loc[ind,'response'] = df.loc[ind,'response'].map({'z': 0, 'm': 1})
# df.loc[~ind,'response'] = df.loc[~ind,'response'].map({'z': 1, 'm': 0})
# df['response'] = df['response'].astype(int)

# # variables:
# df['block_id'] = df['block_id'].astype(int)-1
# df['date'] = pd.to_datetime(df['start_time']).dt.date
# df['block_split'] = (df['trial_nr']>=52).map({False: 1, True: 2}) # FIXME 
# df.loc[df['trial_nr']<=9, 'block_split'] = 0
# df['stimulus'] = df['stimulus'].map({'absent': 0, 'present': 1}).astype(int)
# df['accuracy'] = (df['stimulus'] == df['response']).astype(int)
# df['condition'] = (df['condition']=='boost').astype(int)

# # add pupil
# x = np.array(epochs_p_stim.columns, dtype=float)
# df['pupil_b'] = epochs_p_stim.loc[:,(x>-1.5)&(x<-1)].mean(axis=1).values
# x = np.array(epochs_p_resp.columns, dtype=float)
# df['tpr'] = epochs_p_resp.loc[:,(x>-1)&(x<1.5)].mean(axis=1).values - df['pupil_b']
# # x = np.array(epochs_b_resp.columns, dtype=float)
# # df['blink'] = epochs_b_resp.loc[:,(x>-2)&(x<0)].mean(axis=1).values
# df['blink'] = df['blinks'] > 0
# df['sac'] = df['sacs'] > 0

# # remove:
# counts = df.groupby(['subject_id'])['trial_nr'].count().reset_index()
# accuracies = df.loc[(df['condition']==0) & (df['trial_nr']>9),:].groupby(['subject_id'])['accuracy'].mean().reset_index()
# accuracies['accuracy'] = np.round(accuracies['accuracy']*100)
# blinks_0 = df.loc[(df['condition']==0) & (df['trial_nr']>9),:].groupby(['subject_id'])['blink'].mean().reset_index()
# blinks_1 = df.loc[(df['condition']==1) & (df['trial_nr']>9),:].groupby(['subject_id'])['blink'].mean().reset_index()
# # sacs_0 = df.loc[df['condition']==0,:].groupby(['subject_id'])['repitition'].mean().reset_index()
# # sacs_1 = df.loc[df['condition']==1,:].groupby(['subject_id'])['repitition'].mean().reset_index()

# print(counts)
# print(accuracies)
# print(blinks_0)
# print(blinks_1)
# # print(sacs_0)
# # print(sacs_1)

# counts = counts.loc[counts['trial_nr']<750,:]
# accuracies = accuracies.loc[(accuracies['accuracy']<=60),:]
# blinks_0 = blinks_0.loc[blinks_0['blink']>0.20,:]
# blinks_1 = blinks_1.loc[blinks_1['blink']>0.20,:]

# print(counts)
# print(accuracies)
# print(blinks_0)
# print(blinks_1)

# # exclude subjects and trials:
# exclude = np.array( 
#                     df['subject_id'].isin(counts['subject_id']) |
#                     df['subject_id'].isin(accuracies['subject_id']) |
#                     df['subject_id'].isin(blinks_0['subject_id']) |
#                     df['subject_id'].isin(blinks_1['subject_id']) |
#                     # (df['blink']==1) | 
#                     # df['tpr'].isna() |
#                     (df['rt']<0.3) | 
#                     (df['rt']>2.9)
#                     )
# df = df.loc[~exclude,:]
# epochs_p_stim = epochs_p_stim.loc[~exclude,:]
# epochs_p_resp = epochs_p_resp.loc[~exclude,:]
# # epochs_b_resp = epochs_b_resp.loc[~exclude,:]

# # add session ID:
# # for subj, dd in df.groupby(['subject_id']):
# #     i = 0
# #     for date, d in dd.groupby(['date']):
# #         df.loc[d.index, 'session_id'] = i
# #         i += 1
# df['session_id'] = df.groupby(['subject_id'])['block_id'].apply(pd.cut, bins=3, labels=False)

# # baseline:
# x = np.array(epochs_p_stim.columns, dtype=float)
# baselines = np.atleast_2d(epochs_p_stim.loc[:,(x>-2)&(x<-1)].mean(axis=1)).T
# epochs_p_stim = epochs_p_stim - baselines
# epochs_p_resp = epochs_p_resp - baselines

# # save:

# df.loc[df['trial_nr']>9, 'pupil_b_bin'] = df.loc[df['trial_nr']>9,:].groupby(['subject_id', 'block_id', 'condition'])['pupil_b'].apply(pd.cut, bins=5, labels=False)
# df.loc[df['trial_nr']>9, 'tpr_bin'] = df.loc[df['trial_nr']>9,:].groupby(['subject_id', 'condition'])['tpr'].apply(pd.cut, bins=5, labels=False)
# df.loc[df['condition']==1, 'tpr_bin'] = 6
# df.loc[df['trial_nr']>9,:].to_csv('data.csv')

# # means = df.groupby(['subject_id', 'condition']).mean().reset_index()
# # print(sp.stats.ttest_rel(means.loc[means['condition']==1, 'tpr'], means.loc[means['condition']==0, 'tpr']))

# # groupby = ['subject_id', 'session_id', 'condition']
# # df_res1 = df.groupby(groupby).mean()
# # df_res2 = df.groupby(groupby).apply(sdt)
# # globals().update(locals())
# # df_res = reduce(lambda left, right: pd.merge(left,right, on=groupby), [df_res1, df_res2]).reset_index()
# # for y in ['response', 'accuracy', 'rt', 'd', 'c', 'tpr', 'repitition']:
# #     aovrm = AnovaRM(df_res, y, 'subject_id', within=['session_id', 'condition'], aggregate_func='mean').fit()
# #     print(y)
# #     print(aovrm.summary())

# # shell()


# # from pingouin import mediation_analysis

# # # flip around bias for the (two) liberal subjects:
# # df_res = df.loc[df['condition']==0].groupby(['subject_id']).apply(utils.sdt).reset_index()
# # for subj, d in df_res.loc[df_res['c']<0, :].groupby(['subject_id']):
# #     df.loc[df['subject_id']==subj, 'stimulus'] = df.loc[df['subject_id']==subj, 'stimulus'].map({0:1, 1:0})
# #     df.loc[df['subject_id']==subj, 'response'] = df.loc[df['subject_id']==subj, 'response'].map({0:1, 1:0})



# # mediation_results = []
# # for subj, d in df.loc[df['trial_nr']>9, :].groupby(['subject_id']):
# #     print(subj)
# #     res = mediation_analysis(data=d, x='condition', m='tpr', y='response', covar='stimulus', alpha=0.05,
# #                     seed=42)
# #     res['subj'] = subj
# #     mediation_results.append(res)
# # df_res = pd.concat(mediation_results)


# # df_res['']




# # plot responses:
# fig = utils.plot_pupil_responses(epochs_p_stim.iloc[:,::10], epochs_p_resp.iloc[:,::10], groupby=['subject_id', 'condition'])
# fig.savefig('figs/pupil_r1.pdf')

# # plot RTs:
# median_rts = df.loc[df['condition']==0, :].groupby(['subject_id']).median()['rt']
# fig = plt.figure(figsize=(2,2))
# plt.hist(df.loc[df['condition']==0, 'rt'], bins=25, histtype='stepfilled', color=sns.color_palette()[0])
# plt.title('{} ({} - {})'.format(round(median_rts.mean(),2), 
#                                         round(median_rts.min(),2), 
#                                         round(median_rts.max(),2)))
# plt.xlabel('RT (s)')
# plt.ylabel('Trials (#)')
# sns.despine(trim=True)
# plt.tight_layout()
# fig.savefig('figs/rt_distributions.pdf')

# # plot accuracies:
# mean_accuracies = df.loc[df['condition']==0, :].groupby(['subject_id']).mean()['accuracy']

# df_res = df.loc[df['condition']==0,:].groupby(['subject_id']).mean().reset_index()
# fig = plt.figure(figsize=(1.5,2))
# sns.pointplot(y=df_res['accuracy'], ci=68, color=sns.color_palette()[0])
# sns.stripplot(y=df_res['accuracy'], color=sns.color_palette()[0])
# plt.title('{} ({} - {})'.format(round(mean_accuracies.mean(),2), 
#                                         round(mean_accuracies.min(),2), 
#                                         round(mean_accuracies.max(),2)))
# plt.ylabel('Accuracy (% correct)')
# sns.despine(trim=True)
# plt.tight_layout()
# fig.savefig('figs/correct.pdf')

# # groupby = ['subject_id', 'pupil_b_bin', 'condition']
# # df_res1 = df.groupby(groupby).mean()
# # df_res2 = df.groupby(groupby).apply(sdt)
# # globals().update(locals())
# # df_res = reduce(lambda left, right: pd.merge(left,right, on=groupby), [df_res1, df_res2]).reset_index()
# # for y in ['response', 'accuracy', 'rt', 'd', 'c', 'tpr']:
# #     aovrm = AnovaRM(df_res, y, 'subject_id', within=['pupil_b_bin', 'condition'], aggregate_func='mean').fit()
# #     print(y)
# #     print(aovrm.summary())

# # df = df.loc[df['subject_id']!=34,:]


# # # ANALYSIS ROUND 1:
# # groupby = ['subject_id', 'tpr_bin']
# # # groupby = ['subject_id', 'session_id', 'condition']
# # df_res1 = df.loc[df['trial_nr']>9,:].groupby(groupby).mean()
# # df_res2 = df.loc[df['trial_nr']>9,:].groupby(groupby).apply(utils.sdt)
# # globals().update(locals())
# # df_res = reduce(lambda left, right: pd.merge(left,right, on=groupby), [df_res1, df_res2]).reset_index()
# # df_res['c_abs'] = abs(df_res['c'])
# # df_res['strength'] = 0

# # df_res_d = (df_res.loc[df_res['condition']==1,:].set_index(groupby).droplevel(level=1) - 
# #             df_res.loc[df_res['condition']==0,:].set_index(groupby).droplevel(level=1)).reset_index()
# # df_res_o = ((df_res.loc[df_res['condition']==1,:].set_index(groupby).droplevel(level=1) + 
# #             df_res.loc[df_res['condition']==0,:].set_index(groupby).droplevel(level=1))/2).reset_index()

# # for y in ['rt', 'd', 'c', 'c_abs']:
    
# #     fig = plt.figure()
# #     sns.pointplot(x='tpr_bin', y=y, units='subject_id', ci=68, data=df_res)
    
# #     r = []
# #     for subj, d in df_res.groupby(['subject_id']):
# #         r.append(sp.stats.pearsonr(d.loc[d['tpr_bin']!=6, 'tpr'].values, d.loc[d['tpr_bin']!=6, y].values)[0])
# #     print(sp.stats.wilcoxon(r, np.zeros(len(r))))


# # ANALYSIS ROUND 1:




# df_res, df_res_d, df_res_o = compute_results(df.loc[df['trial_nr']>9,:], groupby=['subject_id', 'condition'])
# # df_res, df_res_d, df_res_o = compute_results(df, groupby=['subject_id', 'condition'])


# n_jobs = 12
# res = Parallel(n_jobs=n_jobs, verbose=1, backend='multiprocessing')(delayed(compute_results)
#                             (df.groupby(['subject_id', 'condition']).sample(frac=1, replace=True), ['subject_id', 'condition'], i) 
#                             for i in range(5000))

# df_res_sem = pd.concat([res[i][0] for i in range(len(res))]).groupby(['subject_id', 'condition']).std().reset_index()
# df_res_d_sem = pd.concat([res[i][1] for i in range(len(res))]).groupby(['subject_id']).std().reset_index()
# df_res_o_sem = pd.concat([res[i][2] for i in range(len(res))]).groupby(['subject_id']).std().reset_index()

# # for y in ['response', 'accuracy', 'rt', 'd', 'c', 'c_abs', 'tpr', 'repitition', 'repitition2']:
# for y in ['c']:
    
#     fig = utils.plot_bars(df_res, y, anova=False)
#     fig.savefig('figs/behavior_{}.pdf'.format(y))

#     fig = plt.figure(figsize=(2,2))
#     r, p = sp.stats.pearsonr(df_res_d['tpr'], df_res_d[y])
#     sns.regplot(x='tpr', y=y, data=df_res_d)
#     plt.axhline(0, color='k', lw=0.5)
#     plt.title('r = {}, p = {}'.format(round(r,2), round(p,3)))
#     plt.xlabel('Δ Pupil response')
#     plt.ylabel('Δ {}'.format(y))
#     sns.despine(trim=True)
#     plt.tight_layout()
#     fig.savefig('figs/behavior_correlation_{}.pdf'.format(y))

#     fig = plt.figure(figsize=(2,2))
#     r, p = sp.stats.pearsonr(df_res_o['c'], y=df_res_d[y])
#     sns.regplot(x=df_res_o['c'], y=df_res_d[y])
#     plt.axhline(0, color='k', lw=0.5)
#     plt.title('r = {}, p = {}'.format(round(r,2), round(p,3)))
#     plt.xlabel('Criterion')
#     plt.ylabel('Δ {}'.format(y))
#     sns.despine(trim=True)
#     plt.tight_layout()
#     fig.savefig('figs/behavior_correlation_2_{}.pdf'.format(y))

#     fig = plt.figure(figsize=(2,2))
#     r, corrrand, p = utils.permutationTest_correlation(df_res.loc[df_res['condition']==0, 'c'].values, 
#                                                         df_res.loc[df_res['condition']==1, 'c'].values, 
#                                                         tail=0, nrand=50000)
#     # r, p = sp.stats.pearsonr(df_res.loc[df_res['condition']==0, 'c'], y=df_res_d[y])
#     plt.errorbar(x=df_res.loc[df_res['condition']==0, 'c'], y=df_res_d[y], 
#                         xerr=df_res_sem.loc[df_res_sem['condition']==0, 'c'], yerr=df_res_d_sem[y],
#                         fmt='o')
#     sns.regplot(x=df_res.loc[df_res['condition']==0, 'c'], y=df_res_d[y], scatter=False)
#     plt.axhline(0, color='k', lw=0.5)
#     plt.title('r = {}, p = {}'.format(round(r,2), round(p,3)))
#     plt.xlabel('Criterion')
#     plt.ylabel('Δ {}'.format(y))
#     sns.despine(trim=True)
#     plt.tight_layout()
#     fig.savefig('figs/behavior_correlation_3_{}.pdf'.format(y))

#     shell()


# plt.close('all')



# # ANALYSIS ROUND 2:

# # # flip around bias for the (two) liberal subjects:
# # df_res = df.loc[df['condition']==0].groupby(['subject_id']).apply(utils.sdt).reset_index()
# # for subj, d in df_res.loc[df_res['c']<0, :].groupby(['subject_id']):
# #     df.loc[df['subject_id']==subj, 'stimulus'] = df.loc[df['subject_id']==subj, 'stimulus'].map({0:1, 1:0})
# #     df.loc[df['subject_id']==subj, 'response'] = df.loc[df['subject_id']==subj, 'response'].map({0:1, 1:0})

# df['block_split'] = df['block_split']-1

# # df_res, df_res_d, df_res_o = compute_results(df, groupby=['subject_id', 'block_split', 'condition'])
# df_res, df_res_d, df_res_o = compute_results(df.loc[df['trial_nr']>9,:], groupby=['subject_id', 'block_split', 'condition'])
# for y in ['response', 'accuracy', 'rt', 'd', 'c', 'c_abs', 'tpr', 'repitition', 'repitition2']:
    
#     # # stats:
#     # t, p = sp.stats.ttest_rel(df_res.loc[df_res['condition']==1, y], 
#     #                             df_res.loc[df_res['condition']==0, y])

#     # ANOVA:
#     aovrm = AnovaRM(df_res, y, 'subject_id', within=['block_split', 'condition'], aggregate_func='mean').fit()
    
#     print()
#     print(y)
#     print(aovrm.summary())
    
#     fig = utils.plot_bars(df_res, y, anova=True, var='block_split')
#     fig.savefig('figs/behavior_block_split_{}.pdf'.format(y))

# # # ANALYSIS ROUND 2:
# # groupby = ['subject_id', 'pupil_b_bin', 'condition']
# # df_res1 = df.groupby(groupby).mean()
# # df_res2 = df.groupby(groupby).apply(sdt)
# # globals().update(locals())
# # df_res = reduce(lambda left, right: pd.merge(left,right, on=groupby), [df_res1, df_res2]).reset_index()
# # df_res['efficiency'] = df_res['rt'] / df_res['accuracy']
# # for y in ['response', 'accuracy', 'rt', 'd', 'c', 'tpr']:
    
# #     # # stats:
# #     # t, p = sp.stats.ttest_rel(df_res.loc[df_res['condition']==1, y], 
# #     #                             df_res.loc[df_res['condition']==0, y])

# #     # ANOVA:
# #     aovrm = AnovaRM(df_res, y, 'subject_id', within=['pupil_b_bin', 'condition'], aggregate_func='mean').fit()
    
# #     print()
# #     print(y)
# #     print(aovrm.summary())
    
# #     # plot bias:
# #     fig = plt.figure(figsize=(1.75,2))
# #     ax = fig.add_subplot(111)
# #     # sns.stripplot(x='condition', y=y, jitter=False, data=df_res)
# #     # sns.pointplot(x='condition', y=y, ci=68, data=df_res)
# #     # for s, d in df_res.groupby(['subject_id']):
# #     #     plt.plot([0,1], [d.loc[d['condition']==0, y], d.loc[d['condition']==1, y]], color='grey', lw=0.75)
# #     sns.pointplot(x='pupil_b_bin', y=y, hue='condition', ci=68, data=df_res)
# #     # plt.xlim(-0.5, 1.5)
# #     # plt.title('t = {}, p = {}'.format(round(t,2), round(p,3)))
# #     ax.get_legend().remove()
# #     plt.ylabel(y)
# #     sns.despine(trim=True)
# #     plt.tight_layout()
# #     fig.savefig('figs/behavior_pupil_b_{}.pdf'.format(y))


# ## SUPPLEMENTS:

# # plot baseline pupil across trials:
# for y in ['rt', 'accuracy', 'pupil_b', 'tpr']:
#     mean = df.groupby(['subject_id', 'trial_nr']).mean().groupby(['trial_nr']).mean()[y]
#     sem = df.groupby(['subject_id', 'trial_nr']).mean().groupby(['trial_nr']).sem()[y]
#     fig = plt.figure(figsize=(2,2))
#     x = np.array(mean.index, dtype=float)
#     plt.axvspan(9,52, color='blue', alpha=0.1)
#     plt.axvspan(52,95, color='red', alpha=0.1)
#     plt.axvline(9, color='k', lw=0.5)
#     plt.axvline(52, color='k', lw=0.5)
#     plt.fill_between(x, mean-sem, mean+sem, alpha=0.2)
#     plt.plot(x, mean)
#     plt.xlabel('Trial (#)')
#     plt.ylabel(y)
#     sns.despine(trim=True)
#     plt.tight_layout()
#     fig.savefig('figs/across_trials_{}.pdf'.format(y))

# # plot behavior across trials:
# mean = df.loc[df['condition']==0,:].groupby(['trial_nr']).apply(utils.sdt).reset_index()
# n_jobs = 64
# n_boot = 250
# sem = pd.concat(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(utils.sdt)
#                                             (d.sample(frac=1, replace=True))
#                                             for _ in range(n_boot)
#                                             for t, d in df.loc[df['condition']==0,:].groupby(['trial_nr']))).reset_index()
# sem['trial_nr'] = np.repeat(np.arange(96), n_boot)
# sem = (sem.groupby(['trial_nr']).quantile(0.84) - sem.groupby(['trial_nr']).quantile(0.16))/2

# for y in ['d', 'c']:
#     fig = plt.figure(figsize=(2,2))
#     x = np.array(mean['trial_nr'], dtype=float)
#     plt.axvspan(9,52, color='blue', alpha=0.1)
#     plt.axvspan(52,95, color='red', alpha=0.1)
#     plt.axvline(9, color='k', lw=0.5)
#     plt.axvline(52, color='k', lw=0.5)
#     plt.fill_between(x, mean[y]-sem[y], mean[y]+sem[y], alpha=0.2)
#     plt.plot(x, mean[y])
#     plt.axvline(9, color='k', lw=0.5)
#     plt.xlabel('Trial (#)')
#     plt.ylabel(y)
#     sns.despine(trim=True)
#     plt.tight_layout()
#     fig.savefig('figs/across_trials_{}.pdf'.format(y))