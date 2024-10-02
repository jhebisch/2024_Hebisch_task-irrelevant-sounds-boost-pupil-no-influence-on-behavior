'''
This script contains functions needed in the preprocessing (0_preprocess) 
and analysis scripts (1_amsterdam_contrast_detection) of experiment 1 
'''

# imports
import datetime
import glob
import itertools
import os
from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import find_peaks
import seaborn as sns
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from tqdm import tqdm
import matplotlib.patches as mpatches
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import multitest
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
import statistics
from pingouin import bayesfactor_ttest

import utils

#from IPython import embed as shell

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
color_noAS = '#DECA59' #'#CBC034' 
color_AS = '#6CA6F0'
color_blend = "blend:#6CA6F0,#702D96"
colorpalette_noAS = "blend:#DECA59,#DE6437"

# sns.set_palette("tab10")

def load_amsterdam_contrast_detection(filename, figs_dir):

    from pyedfread import edf
    import preprocess_pupil

    subj = os.path.basename(filename).split('_')[0]
    ses = os.path.basename(filename).split('_')[1]

    print(filename)

    # load edf and tsv:
    samples, events, messages = edf.pread(filename, trial_marker=b'')
    df_meta_all = pd.read_csv(filename[:-4]+'_events.tsv', sep='\t')
    samples = samples.loc[samples['time']!=0,:].reset_index(drop=True)
    samples = samples.rename({'pa_left': 'pupil'}, axis=1) 
    messages = messages.reset_index(drop=True)

    # load events:
    df_meta = df_meta_all.loc[(df_meta_all['phase']==3)&(df_meta_all['event_type']=='iti'),:].reset_index()
    df_meta['subject_id'] = subj
    df_meta['block_id'] = ses

    # remove doubles:
    df_meta = df_meta.iloc[np.where(df_meta['trial_nr'].diff()!=0)[0]]

    # add timestamps:
    df_meta['time_phase_0'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-0'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000
    df_meta['time_phase_1'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-1'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000
    df_meta['time_phase_2'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-2'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000
    df_meta['time_phase_3'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-3'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000
    df_meta['rt'] = df_meta['time_phase_3'] - df_meta['time_phase_2']

    # datetime
    timestamps = os.path.basename(filename).split('_')[2:]
    timestamps[-1] = timestamps[-1].split('.')[0]
    df_meta['start_time'] = datetime.datetime(*[int(t) for t in timestamps])
    df_meta['morning'] = df_meta['start_time'].dt.time <= datetime.time(13,0,0)
    
    # check if not too many omissions:
    if df_meta.shape[0]<85:
        print('block too short... {} {}'.format(subj, ses))
        return pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])

    # preprocess pupil data:
    fs = int(1/samples['time'].diff().median()*1000)
    params = {'fs':fs, 'lp':10, 'hp':0.01, 'order':3}
    df = preprocess_pupil.preprocess_pupil(samples=samples, events=events, params=params)
    df['time'] = df['time'] / 1000

    # make epochs:
    pupil_measure = 'pupil_int_lp_clean_psc'
    columns = ['subject_id', 'block_id', 'trial_nr', 'condition']

    epochs = utils.make_epochs(df=df, df_meta=df_meta, locking='time_phase_2', start=-2, dur=6, measure=pupil_measure, fs=fs, 
                    baseline=False, b_start=-1, b_dur=1)
    epochs[columns] = df_meta[columns].reset_index(drop=True)
    epochs_p_stim = epochs.set_index(columns)

    epochs = utils.make_epochs(df=df, df_meta=df_meta, locking='time_phase_3', start=-3, dur=6, measure=pupil_measure, fs=fs, 
                        baseline=False, b_start=-1, b_dur=1)
    epochs[columns] = df_meta[columns].reset_index(drop=True)
    epochs_p_resp = epochs.set_index(columns)
    
    # compute blinks and sacs:
    df_meta['blinks_ba_dec'] = [df.loc[(df['time']>(df_meta['time_phase_1'].iloc[i]-0))&(df['time']<df_meta['time_phase_3'].iloc[i]), 'is_blink_new'].mean() 
                for i in range(df_meta.shape[0])]
    df_meta['sacs_ba_dec'] = [df.loc[(df['time']>(df_meta['time_phase_1'].iloc[i]-0))&(df['time']<df_meta['time_phase_3'].iloc[i]), 'is_sac_eyelink'].mean() 
                for i in range(df_meta.shape[0])]
    df_meta['blinks_dec'] = [df.loc[(df['time']>(df_meta['time_phase_2'].iloc[i]-0))&(df['time']<df_meta['time_phase_3'].iloc[i]), 'is_blink_new'].mean() 
                for i in range(df_meta.shape[0])]
    df_meta['sacs_dec'] = [df.loc[(df['time']>(df_meta['time_phase_2'].iloc[i]-0))&(df['time']<df_meta['time_phase_3'].iloc[i]), 'is_sac_eyelink'].mean() 
                for i in range(df_meta.shape[0])]

    # plot result:
    plot = 1
    if plot:
        try:
            fig = utils.plot_blockwise_pupil(df_meta, df, events, pupil_measure)
            fig.savefig(os.path.join(figs_dir, 'preprocess', 'amsterdam_contrast_detection', '{}_{}.pdf'.format(subj, ses)))
        except:
            pass
        
        # # plot evoked responses:
        # x = np.array(epochs_p_stim.columns, dtype=float)
        # baselines = np.atleast_2d(epochs_p_stim.loc[:,(x>-2)&(x<-1)].mean(axis=1)).T
        # epochs_p_stim_b = epochs_p_stim - baselines
        # epochs_p_resp_b = epochs_p_resp - baselines
        # fig = plot_pupil_responses(epochs_p_stim_b.iloc[:,::10], epochs_p_resp_b.iloc[:,::10], groupby=['trial_nr', 'condition'])
        # fig.savefig(os.path.join(figs_dir, 'sessions', '{}_{}_2.pdf'.format(subj, ses)))

        # fig = plt.figure(figsize=(12,6))
        # ax = fig.add_subplot(211)
        # plt.plot(samples['time'], samples['pupil'])
        # plt.plot(samples['time'], samples['pupil_int'])
        # plt.plot(samples['time'], samples['pupil_int_lp'])
        # blinks = events.loc[events['blink']]
        # for i in range(blinks.shape[0]):
        #     plt.axvspan(blinks['start'].iloc[i], blinks['end'].iloc[i], color='r', alpha=0.2)

        # ax = fig.add_subplot(212)
        # plt.plot(samples['time'], samples['pupil_int_lp_psc'])
        # fig.savefig('figs/sessions/{}_{}.jpg'.format(subj, ses))

    # downsample:
    epochs_p_stim = epochs_p_stim.iloc[:,::10]
    epochs_p_resp = epochs_p_resp.iloc[:,::10]

    # return:
    plt.close('all')
    return df_meta, epochs_p_stim, epochs_p_resp

def compute_sequential(df):
    df['repetition_past'] = df['choice']==df['choice'].shift(1) #asks: is the last one the same
    df['repetition_future'] = df['choice']==df['choice'].shift(-1) #asks: is the next one the same
    df['choice_p'] = df['choice'].shift(1)
    df['stimulus_p'] = df['stimulus'].shift(1)
    return df

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
    ind1 = np.array(epochs_p_stim.index.get_level_values('condition')=='boost')
    scalars[ind1] = epochs_p_stim.loc[ind1, (x>-1)&(x<1)].mean(axis=1)

    return pd.DataFrame({'pupil_r_diff': scalars,})

def compute_pupil_scalars(epochs_p_stim, which): 

    # baseline:
    x = epochs_p_stim.columns
    epochs_p_stim = epochs_p_stim - np.atleast_2d(epochs_p_stim.loc[:,(x>-1.5)&(x<-1)].mean(axis=1).values).T
    
    scalars = np.repeat(0, epochs_p_stim.shape[0])
    if which == 'base':
        ind1 = np.array(epochs_p_stim.index.get_level_values('condition')=='normal')
    if which == 'r_stim':
        ind1 = np.array(epochs_p_stim.index.get_level_values('condition')=='boost')
    scalars[ind1] = epochs_p_stim.loc[ind1, (x>-1)&(x<1)].mean(axis=1)

    return pd.DataFrame({'pupil_{}'.format(which): scalars,})

# ---------------------------------------------------
# Figures and analysis functions
# ---------------------------------------------------

# Figure 2
# ---------------------------------------------------

# Fig. 2A 
def plot_normal_pupil(epochs, locking, draw_box=False, shade=False):
    
    means = epochs.groupby(['subject_id', 'condition', 'tpr_bin']).mean().groupby(['condition', 'tpr_bin']).mean()
    sems = epochs.groupby(['subject_id', 'condition', 'tpr_bin']).mean().groupby(['condition', 'tpr_bin']).sem()
    means = means.loc[means.index.get_level_values(0)=='normal']
    sems = sems.loc[sems.index.get_level_values(0)=='normal']

    
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(111)
    x = np.array(means.columns, dtype=float)
    if not shade==False:
        plt.axvspan(-0.5,0, color='grey', alpha=0.1)
        plt.axvspan(-3.5,-3, color='grey', alpha=0.2)
    if draw_box:
        plt.axvspan(-0.5,1.5, color='grey', alpha=0.15)
    plt.axvline(0, color='k', ls='--', linewidth=0.5)
   
    for i in means.index.get_level_values(1):
        plt.fill_between(x, means.iloc[int(i)]-sems.iloc[int(i)], 
                            means.iloc[int(i)]+sems.iloc[int(i)], alpha=0.25, color=sns.color_palette(colorpalette_noAS,(len(means.index.get_level_values(1))+1))[int(i)+1])
        plt.plot(x, means.iloc[int(i)], color=sns.color_palette(colorpalette_noAS,(len(means.index.get_level_values(1))+1))[int(i)+1])

    # plt.xticks([-4,-3,-2,-1,0,1,2])
    # plt.yticks([-2,-1,0,1])
    if locking == 'dphase':
        plt.xlabel('Time from decision interval onset (s)')
    if locking == 'resp':
        plt.xlabel('Time from button press (s)')

    plt.ylabel('Pupil response (% change)')
    # plt.legend()

    print(means.loc[:, (x>-0.5)&(x<1.5)].mean(axis=1))

    sns.despine(trim=True)
    plt.tight_layout()

    return fig

# Fig. 2B
def plot_pupil_responses(epochs, draw_box=False):
    
    means = epochs.groupby(['subject_id', 'condition']).mean()
    for (subj), m in means.groupby(['subject_id']):
        means.loc[m.index[0],:] = means.loc[m.index[0],:] - means.loc[m.index[1],:]
    sems = means.groupby(['condition']).sem()
    means = means.groupby(['condition']).mean()

    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(111)
    x = np.array(means.columns, dtype=float)
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.axvspan(-1, 1, color='grey', alpha=0.15)
    x2 = range(-1,4)
    y = [-0.5, -0.5, -0.5, -0.5, -0.5]
    plt.plot(x2, y, color='grey', lw=2, ls='-', solid_capstyle='round')
    plt.fill_between(x, means.iloc[0]-sems.iloc[0], 
                        means.iloc[0]+sems.iloc[0], 
                        color=sns.color_palette(color_blend,1)[0], alpha=0.2)
    plt.plot(x, means.iloc[0], color=sns.color_palette(color_blend,1)[0], label=means.iloc[0].name)
    # plt.axhline(0, color='black', lw=0.5)
    plt.xlabel('Time from decision interval onset (s)')
    plt.ylabel('Δ Pupil response (% change)')
    # plt.legend()
    # plt.axvspan(-0.5, 0, color='grey', alpha=0.2)
    sns.despine(trim=True)
    plt.tight_layout()

    return fig

# Figures 3 +S4 +S5
# ---------------------------------------------------
# function for fig S4B split by block type:
def plot_split_tpr_behav_cor(df_res, m):
    fig = plt.figure(figsize=(2,2))
    plt_nr = 1
    colors = [color_noAS, "#E06138"]
    labels = ['rare', 'frequent']
    # patch = []

    ax = fig.add_subplot(1,1,plt_nr)
    for z in [0,1]:
        means = df_res.loc[df_res['block_type']==z].groupby(['tpr_bin']).mean()
        sems = df_res.loc[df_res['block_type']==z].groupby(['tpr_bin']).sem()

        r = []
        for subj, d in df_res.loc[df_res['block_type']==z].groupby(['subject_id']):
            r.append(sp.stats.pearsonr(d['tpr_c'],d[m])[0])
        t, p = sp.stats.ttest_1samp(r, 0)
        plt.errorbar(x=means['tpr_c'], y=means[m], yerr=sems[m], fmt='o', color=colors[z], alpha=0.7)
        # patch.append(mpatches.Patch(color=colors[z], label= labels[z]))
        
        if p < 0.05:
            sns.regplot(x=means['tpr_c'], y=means[m], scatter=False, ci=None, color=colors[z])
        s = round(p,3)
        if s == 0.0:
            txt = ', p < 0.001'
        else:
            txt = ', p = {}'.format(s)
        if m == 'c_abs':
            if s<0.05:
                plt.text(x=-13, y=0.24-(z*0.07), s='$r_{{{}}}=$'.format(labels[z])+str(round(statistics.mean(r),3))+r'$\bf{{{}}}$'.format(txt), size=7, va='center', ha='left')
            else:
                plt.text(x=-13, y=0.24-(z*0.07), s='$r_{{{}}}=$'.format(labels[z])+str(round(statistics.mean(r),3))+txt, size=7, va='center', ha='left')
            plt.ylim(0.13,0.8)
        if m == 'd':
            if s<0.05:
                plt.text(x=-13, y=1.27-(z*0.08), s='$r_{{{}}}=$'.format(labels[z])+str(round(statistics.mean(r),3))+r'$\bf{{{}}}$'.format(txt), size=7, va='center', ha='left')
            else:
                plt.text(x=-13, y=1.27-(z*0.08), s='$r_{{{}}}=$'.format(labels[z])+str(round(statistics.mean(r),3))+txt, size=7, va='center', ha='left')
            plt.ylim(1.14,1.9)
        if m == 'rt':
            if s<0.05:
                plt.text(x=-13, y=1.07-(z*0.03), s='$r_{{{}}}=$'.format(labels[z])+str(round(statistics.mean(r),3))+r'$\bf{{{}}}$'.format(txt), size=7, va='center', ha='left')
            else:
                plt.text(x=-13, y=1.07-(z*0.03), s='$r_{{{}}}=$'.format(labels[z])+str(round(statistics.mean(r),3))+txt, size=7, va='center', ha='left')
            plt.ylim(1.02,1.34)

    # plt.title('p = {}'.format(round(p,3)))
    # fig.legend(title= 'Visual stimulus', title_fontsize = 6, handles=patch, loc='upper right', bbox_to_anchor=(1.16, 0.95)) #, loc='upper left', bbox_to_anchor=(1, 1))    
    plt.xlabel('Pupil response\n(% signal change)')
    if m == 'c_abs':
        # plt.title('Bias', fontweight='bold')
        plt.ylabel('| Criterion c |')
    if m == 'c':
        # plt.title('Bias', fontweight='bold')
        plt.ylabel('Criterion c')
    if m == 'd':
        # plt.title('Performance', fontweight='bold')
        plt.ylabel("Sensitivity d'")
    if m == 'rt':
        # plt.title('Reaction time', fontweight='bold')
        plt.ylabel('Reaction time')
    if m == 'correct':
        plt.ylabel('Accuracy')
    plt_nr+=1
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

# fig 3B
def plot_compare(df_res, y, comp_x1='pupil_base', comp_x2='pupil_r_stim'):
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(1,1,1)
    means = df_res.groupby(['condition']).mean().reset_index()
    sems = df_res.groupby(['condition']).sem().reset_index()
    plt.errorbar(x=means.loc[means['condition']==0,comp_x1], y=means.loc[means['condition']==0,y], xerr=sems.loc[means['condition']==0,comp_x1], yerr=sems.loc[means['condition']==0,y], fmt='o', color=color_noAS, alpha=0.7)
    plt.errorbar(x=means.loc[means['condition']==1,comp_x2], y=means.loc[means['condition']==1,y], xerr=sems.loc[means['condition']==1, comp_x2], yerr=sems.loc[means['condition']==1,y], fmt='o', color=color_AS, alpha=0.7)
    plt.xlabel('Δ Pupil response\n(% signal change)')
    if y == 'c_abs_diff':
        plt.ylabel('Δ | Criterion c |')
    if y == 'c_abs':
        plt.ylabel('| Criterion c |')
    if y == 'c':
        plt.ylabel('Criterion c')
    if y == 'd':
        plt.ylabel("Sensitivity d'")
    if y == 'rt':
        plt.ylabel('Reaction time')
    if y == 'correct':
        plt.ylabel('Accuracy')
    plt.xlim(-3,6)
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

# fig 3C
def plot_se_behav(df_res, m, data_set, split=True):
    p_values = []
    bf_values = []
    t_values = []
    if split == True:
        fig = plt.figure(figsize=(2,2.25)) #2,2.5
        ax = fig.add_subplot() #121)
        for c in [0,1]:
            y = df_res.loc[(df_res['condition']==1) & (df_res['block_type']==c), m].values-df_res.loc[(df_res['condition']==0) & (df_res['block_type']==c), m].values
            sns.stripplot(x=c, y=y, 
                        color=sns.color_palette(color_blend,2)[c], ax=ax, linewidth=0.2, alpha=0.7, edgecolor=sns.color_palette(color_blend,2)[c])
            ax.hlines(y=y.mean(), xmin=(c)-0.4, xmax=(c)+0.4, zorder=10, colors='k')

            p_values.append(sp.stats.ttest_rel(df_res.loc[(df_res['condition']==1) & (df_res['block_type']==c), m].values,
                            df_res.loc[(df_res['condition']==0) & (df_res['block_type']==c), m].values)[1])
            t_values.append(sp.stats.ttest_rel(df_res.loc[(df_res['condition']==1) & (df_res['block_type']==c), m].values,
                            df_res.loc[(df_res['condition']==0) & (df_res['block_type']==c), m].values)[0])
            bf_values.append(bayesfactor_ttest(t_values[c], len(df_res['subject_id'].unique()), paired=True))
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            if m == 'pupil_r_diff':
                plt.text(x=c+0.2, y=0.9, s='$BF_{{{0:}}}={1: .3g}$'.format('10',round(bf_values[c],3)), size=5, rotation=90, transform=trans)
            else:
                plt.text(x=c+0.1, y=0.9, s='$BF_{{{}}}={}$'.format('10',round(bf_values[c],3)), size=5, rotation=45, transform=trans)
        ax.set_xticklabels(['rare', 'frequent'])
        ax.set_xlabel('Visual stimulus')

    elif split == False:
        df_res1 = df_res.loc[(df_res['condition']==1)].groupby('subject_id').mean()
        df_res2 = df_res.loc[(df_res['condition']==0)].groupby('subject_id').mean()
        fig = plt.figure(figsize=(2,2)) 
        ax = fig.add_subplot() 
        y = df_res1[m].values-df_res2[m].values
        print('{} {}: mean difference'.format(data_set, m), y.mean(), ', sd difference', y.std())
        sns.stripplot(x=1, y=y, 
                    color=sns.color_palette(color_blend,2)[0], ax=ax, linewidth=0.2, alpha=0.7, edgecolor=sns.color_palette(color_blend,2)[0])
        ax.hlines(y=y.mean(), xmin=(0)-0.4, xmax=(0)+0.4, zorder=10, colors='k')
        ax.set_xlim([-1,1])

        # p_values.append(sp.stats.ttest_rel(df_res.loc[(df_res['condition']==1), m].values,
        #                 df_res.loc[(df_res['condition']==0), m].values)[1])
        # t_values.append(sp.stats.ttest_rel(df_res.loc[(df_res['condition']==1), m].values,
        #                 df_res.loc[(df_res['condition']==0), m].values)[0])
        p_values.append(sp.stats.ttest_rel(df_res1[m].values,
                        df_res2[m].values)[1])
        t_values.append(sp.stats.ttest_rel(df_res1[m].values,
                        df_res2[m].values)[0])
        bf_values.append(bayesfactor_ttest(t_values[-1], len(df_res['subject_id'].unique()), paired=True))
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        if (m == 'c') or (m == 'rt') or (m == 'd'):
            if (m == 'pupil_r_diff'):
                plt.text(x=0.2, y=0.9, s='$BF_{{{0:}}}={1: .3g}$'.format('10',round(bf_values[0],3)), size=5, rotation=90, transform=trans)
            if (m == 'rt'):
                plt.text(x=0.2, y=0.7, s='$BF_{{{0:}}}={1: .3g}$'.format('10',round(bf_values[0],3)), size=5, rotation=90, transform=trans)
            else:
                plt.text(x=0.2, y=0.9, s='$BF_{{{}}}={}$'.format('10',round(bf_values[0],3)), size=5, rotation=90, transform=trans)
        ax.get_xaxis().set_visible(False)

    for i in range(len(bf_values)): 
        print('{} {} BF:'.format(data_set, m), '{0: .3g},'.format(round(bf_values[i]), 3), bf_values[i])
        print('{} {} t,p:'.format(data_set, m), round(t_values[i], 3), round(p_values[i], 3))
    plt.axhline(0, color='black', lw=0.5)
    # ax.set_xlim([-1,1])
    if m == 'pupil_r_diff':
        plt.ylabel('Δ Pupil response (% change)')
    if m == 'c_abs':
        ax.set_ylabel('Δ | Criterion c |')
    if m == 'c':
        ax.set_ylabel('Δ Criterion c')
    if m == 'd':
        ax.set_ylabel("Δ Sensitivity d'")
    if m == 'rt':
        if split==False:
            plt.ylim(-0.4, 0.2)
        plt.ylabel('Δ Reaction time')
    if m == 'correct':
        plt.ylabel('Δ Accuracy')

    sns.despine(trim=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    return fig

# Figure 5
# ------------------------------------------
def baseline_interaction(m,n,y,z):
    # model = AnovaRM(data=n, depvar=m, subject='subject_id', within=['condition', 'pupil_b_bin', 'block_type'])
    n = n.groupby(['subject_id', 'condition', 'pupil_b_bin']).mean().reset_index()
    model = AnovaRM(data=n.loc[n['condition']==1], depvar=m, subject='subject_id', within=['pupil_b_bin'])

    # model = AnovaRM(data=n, depvar=m, subject='subject_id', within=['pupil_b_e_bin'])
    res = model.fit()
    print(model.fit())

    fig = plt.figure(figsize=(2.3,2.3))
    ax = fig.add_subplot() #121)
    means = n.groupby(['pupil_b_bin']).mean()
    sems = n.groupby(['pupil_b_bin']).sem()
    plt.errorbar(x=means['pupil_b'], y=means[m], xerr=sems['pupil_b'], yerr=sems[m], fmt='o', color=color_AS, alpha=0.7)
    # sns.lineplot(data=n, x='pupil_b_bin', y=m, errorbar='se', palette = [color_AS])
    # sns.lineplot(data=df_res, x='pupil_b_bin', y='c_abs', hue='condition', errorbar='se', palette = [color_noAS, color_AS]) 
    
    f = res.anova_table["F Value"][0]
    df1 = res.anova_table["Num DF"][0]
    df2 = res.anova_table["Den DF"][0]
    s = res.anova_table["Pr > F"][0]
    if s < 0.0001:
        txt = ', p < 0.001'
    else:
        txt = ', p = {}'.format(round(s,3))
    if s < 0.05:
        plt.text(x=-20, y=y, s='$F_{{({},{})}}=$'.format(df1, df2)+str(round(f,3))+r'$\bf{{{}}}$'.format(txt), size=7, va='center', ha='left')
    else:
        plt.text(x=-20, y=y, s='$F_{{({},{})}}=$'.format(df1, df2)+str(round(f,3))+txt, size=7, va='center', ha='left')
    plt.ylim(z[0],z[1])
    if m == 'pupil_r_diff':
        plt.ylabel('Δ Pupil response (% change)')
    if m == 'c_abs':
        ax.set_title('Bias', fontweight='bold')
        ax.set_ylabel('| Criterion c |')
    if m == 'c_abs_diff':
        ax.set_ylabel('Δ | Criterion c |')
    if m == 'c':
        ax.set_title('Bias',fontweight='bold')
        ax.set_ylabel('Criterion c')
    if m == 'd':
        ax.set_title('Performance',fontweight='bold')
        ax.set_ylabel("Sensitivity d'")
    if m == 'rt':
        ax.set_title('Reaction time',fontweight='bold')
        plt.ylabel('Reaction time')
    if m == 'correct':
        plt.ylabel('Accuracy')
    plt.xlabel('Baseline pupil size (% w.r.t. mean)')
    # plt.xticks([0,1,2])
    # leg_handles = ax.get_legend_handles_labels()[0]
    # ax.legend(leg_handles,['no AS', 'AS'], title= 'Condition', title_fontsize = 6) # , loc='upper right', bbox_to_anchor=(1.16, 0.95)) 
    plt.legend([], [], frameon=False)

    sns.despine(trim=True)
    plt.tight_layout()

    return fig

# Figure S1
# ------------------------------------------
def plot_means(df_res, measure, data_set, split=False):
    # df_res = utils.compute_results(df=df.loc[:,:], groupby=['subject_id'])
    # means = df_res[measure].mean()
    # sems = df_res[measure].sem()
    print('{} mean {}'.format(data_set, measure), df_res[measure].mean())
    print('{} sem {}'.format(data_set, measure), df_res[measure].sem())

    if split == False:
        fig = plt.figure(figsize=(1.3,2))
        ax = fig.add_subplot() 
        sns.stripplot(x=0, y=df_res[measure], color='grey', ax=ax, linewidth=0.2, alpha=0.7) # , edgecolor=sns.color_palette(color_blend,2)[c])
        ax.hlines(y=df_res[measure].mean(), xmin=-0.4, xmax=0.4, zorder=10, colors='k')
        ax.set_xlim([-1,1])
        ax.get_xaxis().set_visible(False)
    elif split == True:
        fig = plt.figure(figsize=(2,2))
        ax = fig.add_subplot() 
        sns.stripplot(x=0, y=df_res.loc[df_res['block_type']==0][measure], color='grey', ax=ax, linewidth=0.2, alpha=0.7) # , edgecolor=sns.color_palette(color_blend,2)[c])
        sns.stripplot(x=1, y=df_res.loc[df_res['block_type']==1][measure], color='grey', ax=ax, linewidth=0.2, alpha=0.7) # , edgecolor=sns.color_palette(color_blend,2)[c])
        ax.set_xticklabels(['rare', 'frequent'])
        ax.hlines(y=df_res.loc[df_res['block_type']==0][measure].mean(), xmin=-0.4, xmax=0.4, zorder=10, colors='k')
        ax.hlines(y=df_res.loc[df_res['block_type']==1][measure].mean(), xmin=0.6, xmax=1.4, zorder=10, colors='k')

    if measure == 'pupil_r_diff':
        plt.ylabel('Pupil response (% change)')
    if measure == 'c_abs':
        ax.set_ylabel('| Criterion c |')
    if measure == 'c':
        ax.set_ylabel('Criterion c')
    if measure == 'd':
        ax.set_ylabel("Sensitivity d'")
    if measure == 'rt':
        plt.ylabel('Reaction time')
    if measure == 'correct':
        plt.ylabel('Accuracy')

    sns.despine(trim=True)
    plt.tight_layout()

    return fig

# Figure S2
# ------------------------------------------
def plot_across_trials(y, means, sems, trial_cutoff):
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot()
    plt.axvline(trial_cutoff, color='k', lw=0.5)
    color_choices = [color_noAS,color_AS]
    for i in range(len(means)):
        x = np.array(means[i]['trial_nr'])
        window = 10
        mean = means[i].rolling(window=window, center=True, min_periods=1).mean()
        sem = sems[i].rolling(window=window, center=True, min_periods=1).mean()
        if y == 'pupil_r_diff':
            if int(mean['condition'].mean()) == 1:
                plt.fill_between(x, mean[y]-sem[y], mean[y]+sem[y], alpha=0.2, color=color_choices[int(mean['condition'].mean())])
                # plt.plot(x, mean[y], color=color_choices[int(mean['condition'].mean())])
                plt.scatter(x, mean[y], color=color_choices[int(mean['condition'].mean())], s=2.7, marker='o', lw=0.05, alpha=0.7) # , edgecolors='white')
        elif (y == 'rt') | (y == 'correct') | (y == 'd') | (y == 'c') | (y == 'c_abs') | (y == 'pupil_b') | (y == 'tpr'):
            plt.fill_between(x, mean[y]-sem[y], mean[y]+sem[y], alpha=0.2, color=color_choices[int(mean['condition'].mean())])
            plt.scatter(x, mean[y], color=color_choices[int(mean['condition'].mean())], s=2.7, marker='o', lw=0.05, alpha=0.7)
            # plt.plot(x, mean[y], color=color_choices[int(mean['condition'].mean())])
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
    return fig

# Figure S6
# ------------------------------------------
def pupil_tis_resp_behav_cor(df_res, indices, r_place):
    fig = plt.figure(figsize=((2*len(indices)),2))
    plt_nr = 1
    for m in indices:
        ax = fig.add_subplot(1,len(indices),plt_nr)
        means = df_res.groupby(['pupil_r_diff_bin']).mean()
        sems = df_res.groupby(['pupil_r_diff_bin']).sem()

        model = smf.mixedlm("c_abs_diff ~ pupil_r_diff", groups=df_res["subject_id"], re_formula="~pupil_r_diff", data=df_res) # re_formula="0", #TODO check if correct
        result = model.fit()

        r = []
        for subj, d in df_res.groupby(['subject_id']):
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
            plt.text(x=r_place, y=max(means[m])+0.05, s='r='+str(round(statistics.mean(r),3))+txt, size=7, va='center', ha='center')
        plt.xlabel('Δ Pupil response\n(% signal change)')
        if m == 'c_abs_diff':
            # plt.title('Bias')
            plt.ylabel('Δ | Criterion c |')
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
        # plt_nr+=1
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

