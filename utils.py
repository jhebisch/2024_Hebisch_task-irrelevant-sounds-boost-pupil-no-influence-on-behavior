import os, glob, datetime
import itertools
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
import statistics
import statsmodels.api as sm
import statsmodels.formula.api as smf

from joblib import Memory

# memory = Memory(os.path.expanduser('cache'), verbose=0)

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
# sns.set_palette("tab10")
color_noAS = '#DECA59' #'#CBC034' 
color_blend = "blend:#6CA6F0,#702D96"

def bin_correct_tpr(df, plus_correct='rt', trial_cutoff=9):
    import statsmodels.formula.api as smf
    model = smf.ols(formula='tpr ~ pupil_b + {}'.format(plus_correct), data=df.loc[df['trial_nr']>trial_cutoff,:]).fit()
    df.loc[df['trial_nr']>trial_cutoff, 'tpr_c'] = model.resid + df.loc[df['trial_nr']>trial_cutoff, 'tpr'].mean()

    return df

def sdt(df, stim_column, response_column):

    # counts:
    n_hit = ((df[stim_column]==1)&(df[response_column]==1)).sum()
    n_miss = ((df[stim_column]==1)&(df[response_column]==0)).sum()
    n_fa = ((df[stim_column]==0)&(df[response_column]==1)).sum()
    n_cr = ((df[stim_column]==0)&(df[response_column]==0)).sum()
    
    # regularize:
    n_hit += 0.5
    n_miss += 0.5
    n_fa += 0.5
    n_cr += 0.5
    
    # rates:
    hit_rate = n_hit / (n_hit + n_miss)
    fa_rate = n_fa / (n_fa + n_cr)
    
    # z-score:
    hit_rate_z = sp.stats.norm.isf(1-hit_rate)
    fa_rate_z = sp.stats.norm.isf(1-fa_rate)
    
    # measures:
    d = hit_rate_z - fa_rate_z
    c = -(hit_rate_z + fa_rate_z) / 2

    return pd.DataFrame({'d':[d], 'c':[c], 'hr':[hit_rate], 'far':[fa_rate]})

def make_epochs(df, df_meta, locking, start, dur, measure, fs, baseline=False, b_start=-1, b_dur=1):

    # make sure we start with index 0:
    df_meta = df_meta.reset_index(drop=True)

    # locking_inds = np.array(df['time'].searchsorted(df_meta.loc[~df_meta[locking].isna(), locking]).ravel())
    locking_inds = np.array(df['time'].searchsorted(df_meta[locking]).ravel())
    # locking_inds = np.array([find_nearest(np.array(df['time']), t) for t in df_meta[locking]])
    
    # print(locking_inds)

    start_inds = locking_inds + int(start/(1/fs))
    end_inds = start_inds + int(dur/(1/fs)) - 1
    start_inds_b = locking_inds + int(b_start/(1/fs))
    end_inds_b = start_inds_b + int(b_dur/(1/fs))
    
    epochs = []
    for s, e, sb, eb in zip(start_inds, end_inds, start_inds_b, end_inds_b):
        epoch = np.array(df.loc[s:e, measure]) 
        if baseline:
            epoch = epoch - np.array(df.loc[sb:eb,measure]).mean()
        if s < 0:
            epoch = np.concatenate((np.repeat(np.NaN, abs(s)), epoch))
        epochs.append(epoch)
    epochs = pd.DataFrame(epochs)
    epochs.columns = np.arange(start, start+dur, 1/fs).round(5)
    if df_meta[locking].isna().sum() > 0:
        epochs.loc[df_meta[locking].isna(),:] = np.NaN

    return epochs

def compute_results(df, groupby, iteration=0): 

    # groupby = ['subject_id', 'session_id', 'condition']
    df_res1 = df.groupby(groupby).mean(numeric_only=True) # mean() in pandas version 1.5.2
    df_res1['choice_abs'] = abs(df_res1['choice']-0.5)
    df_res2 = df.groupby(groupby).apply(sdt, stim_column='stimulus', response_column='choice')
    df_res = reduce(lambda left, right: pd.merge(left,right, on=groupby), [df_res1, df_res2]).reset_index()
    df_res['c_abs'] = abs(df_res['c']) #TODO move to sdt function
    df_res['strength'] = 0

    # df_res_d = (df_res.loc[df_res['condition']==1,:].set_index(groupby).droplevel(level=1) - 
    #             df_res.loc[df_res['condition']==0,:].set_index(groupby).droplevel(level=1)).reset_index()
    # df_res_o = ((df_res.loc[df_res['condition']==1,:].set_index(groupby).droplevel(level=1) + 
    #             df_res.loc[df_res['condition']==0,:].set_index(groupby).droplevel(level=1))/2).reset_index()
    
    df_res['iteration'] = iteration
    # df_res_o['iteration'] = iteration
    # df_res_d['iteration'] = iteration

    return df_res

def compute_sequential(df):
    df['repitition'] = df['response']==df['response'].shift(1) #repetition past
    df['repitition2'] = df['response']==df['response'].shift(-1) #repetition future
    df['choice_p'] = df['choice'].shift(1)
    df['stimulus_p'] = df['stimulus'].shift(1)
    return df

## Plotting functions

def plot_blockwise_pupil(df_meta, df, events, pupil_measure):

    blinks = events.loc[events['blink']]
    blinks['start'] = blinks['start'] / 1000
    blinks['end'] = blinks['end'] / 1000

    import matplotlib.gridspec as gridspec
    
    start = float(df['time'].iloc[0])
    end = float(df['time'].iloc[-1])

    # plot:
    fig = plt.figure(figsize=(7.5, 2.5))
    gs = gridspec.GridSpec(5, 1)

    try:
        ax = fig.add_subplot(gs[0, :])
        for t, d in df_meta.groupby(['trial_nr']):
            ax.axvspan((float(d['time_phase_1'])-start)/60, (float(d['time_phase_2'])-start)/60, color='grey', lw=0)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.xlim(0, (end-start)/60)
        plt.ylim(0,2)

        ax = fig.add_subplot(gs[1, :])
        for t, d in df_meta.loc[df_meta['condition']=='boost',:].groupby(['trial_nr']):
            ax.axvspan((float(d['time_phase_1'])-1-start)/60, (float(d['time_phase_2'])-start)/60, color='red', lw=0)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.xlim(0, (end-start)/60)
        plt.ylim(0,2)
    except:
        pass

    ax = fig.add_subplot(gs[2:5, :])
    for i in range(blinks.shape[0]):
        ax.axvspan((blinks['start'].iloc[i]-start)/60, (blinks['end'].iloc[i]-start)/60, color='red', alpha=0.2, lw=0)
    ax.plot((df['time'].iloc[::10]-start)/60, df[pupil_measure].iloc[::10])
    plt.xlim(0, (end-start)/60)
    plt.xlabel('Time (min)')

    sns.despine(trim=False)
    plt.tight_layout()

    return fig

def plot_pupil_responses(epochs_p_stim, epochs_p_resp, groupby=['subject_id', 'condition'], ylim=None):

    fig = plt.figure(figsize=(3.5,3.5))
    ax = fig.add_subplot(221)
    means = epochs_p_stim.groupby(groupby).mean().groupby(['condition']).mean()
    sems = epochs_p_stim.groupby(groupby).mean().groupby(['condition']).sem()
    x = np.array(means.columns, dtype=float)
    plt.fill_between(x, means.iloc[0]-sems.iloc[0], means.iloc[0]+sems.iloc[0], color=sns.color_palette()[1], alpha=0.2)
    plt.plot(x, means.iloc[0], color=sns.color_palette()[1],    ls='-', label='boost')
    plt.fill_between(x, means.iloc[1]-sems.iloc[1], means.iloc[1]+sems.iloc[1], color=sns.color_palette()[0], alpha=0.2)
    plt.plot(x, means.iloc[1], color=sns.color_palette()[0], ls='-', label='normal')
    plt.axvline(0, color='k', ls='--')
    plt.legend(loc=4)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Time from stimulus (s)')
    plt.ylabel('Pupil response (% change)')
        
    ax = fig.add_subplot(222)
    means = epochs_p_stim.groupby(groupby).mean().groupby(['condition']).mean()
    sems = epochs_p_stim.groupby(groupby).mean().groupby(['condition']).sem()
    x = np.array(means.columns, dtype=float)
    mean = (means.iloc[0]-means.iloc[1])/2
    sem = (sems.iloc[0]+sems.iloc[1])/2
    plt.fill_between(x, mean-sem, mean+sem, color='grey', alpha=0.2)
    plt.plot(x, mean, color='grey',    ls='-', label='difference')
    plt.axvline(0, color='k', ls='--')
    plt.legend(loc=4)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Time from stimulus (s)')
    plt.ylabel('Pupil response (% change)')

    ax = fig.add_subplot(223)
    means = epochs_p_resp.groupby(groupby).mean().groupby(['condition']).mean()
    sems = epochs_p_resp.groupby(groupby).mean().groupby(['condition']).sem()
    x = np.array(means.columns, dtype=float)
    plt.fill_between(x, means.iloc[0]-sems.iloc[0], means.iloc[0]+sems.iloc[0], color=sns.color_palette()[1], alpha=0.2)
    plt.plot(x, means.iloc[0], color=sns.color_palette()[1],    ls='-', label='boost')
    plt.fill_between(x, means.iloc[1]-sems.iloc[1], means.iloc[1]+sems.iloc[1], color=sns.color_palette()[0], alpha=0.2)
    plt.plot(x, means.iloc[1], color=sns.color_palette()[0], ls='-', label='normal')
    plt.axvspan(-1, 1.5, color='grey', alpha=0.1)
    plt.axvline(0, color='k', ls='--')
    plt.legend(loc=4)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Time from choice (s)')
    plt.ylabel('Pupil response (% change)')

    ax = fig.add_subplot(224)
    means = epochs_p_resp.groupby(groupby).mean().groupby(['condition']).mean()
    sems = epochs_p_resp.groupby(groupby).mean().groupby(['condition']).sem()
    x = np.array(means.columns, dtype=float)
    mean = (means.iloc[0]-means.iloc[1])/2
    sem = (sems.iloc[0]+sems.iloc[1])/2
    plt.fill_between(x, mean-sem, mean+sem, color='grey', alpha=0.2)
    plt.plot(x, mean, color='grey',    ls='-', label='difference')
    plt.axvline(0, color='k', ls='--')
    plt.legend(loc=4)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Time from choice (s)')
    plt.ylabel('Pupil response (% change)')

    sns.despine(trim=True)
    plt.tight_layout()
    return fig

def plot_tpr_behav_cor(df, groupby, indices, r_place=0): #plotting function for fig 2
    fig = plt.figure(figsize=((2*len(indices)),2))
    plt_nr = 1
    df_res = compute_results(df=df.loc[:,:], groupby=groupby)
    df_res = df_res.loc[df_res['condition']==0,:]
    for m in indices:
        ax = fig.add_subplot(1,len(indices),plt_nr)
        means = df_res.groupby(['tpr_bin']).mean()
        sems = df_res.groupby(['tpr_bin']).sem()

        model = smf.mixedlm("c_abs ~ tpr_c", groups=df_res["subject_id"], re_formula="~tpr_c", data=df_res) # re_formula="0", #TODO check if correct
        result = model.fit()

        r = []
        for subj, d in df_res.groupby(['subject_id']):
            r.append(sp.stats.pearsonr(d['tpr_c'],d[m])[0])
        t, p = sp.stats.ttest_1samp(r, 0)
        plt.errorbar(x=means['tpr_c'], y=means[m], yerr=sems[m], fmt='o', color=color_noAS, alpha=0.7)
        if p < 0.05:
            sns.regplot(x=means['tpr_c'], y=means[m], scatter=False, ci=None, color='k')
        s = round(p,3)
        if s == 0.0:
            txt = ' p < 0.001'
        else:
            txt = ' p = {}'.format(s)
        if m=='rt':
            if s<0.05:
                plt.text(x=r_place, y=max(means[m])+0.04, s='r='+str(round(statistics.mean(r),3))+','+r'$\bf{{{}}}$'.format(txt), size=7, va='center', ha='center')
            else: 
                plt.text(x=r_place, y=max(means[m])+0.04, s='r='+str(round(statistics.mean(r),3))+','+txt, size=7, va='center', ha='center')
        else:        
            if s<0.05:
                plt.text(x=r_place, y=max(means[m])+0.05, s='r='+str(round(statistics.mean(r),3))+','+r'$\bf{{{}}}$'.format(txt), size=7, va='center', ha='center')
            else:
                plt.text(x=r_place, y=max(means[m])+0.05, s='r='+str(round(statistics.mean(r),3))+','+txt, size=7, va='center', ha='center')
        plt.xlabel('Pupil response\n(% signal change)')
        if m == 'c_abs':
            # plt.title('Bias')
            plt.ylabel('| Criterion c |')
        if m == 'c':
            # plt.title('Bias')
            plt.ylabel('Criterion c')
        if m == 'd':
            # plt.title('Performance')
            plt.ylabel("Sensitivity d'")
        if m == 'rt':
            # plt.title('Reaction time')
            plt.ylabel('Reaction time')
        if m == 'correct':
            plt.ylabel('Accuracy')
        plt_nr+=1
    sns.despine(trim=True)
    plt.tight_layout()
    return fig, result, t

