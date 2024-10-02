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
import utils
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
    # 'lines.linewidth': 0.8,
    'ytick.color':'Black',} )
sns.plotting_context()
color_noAS = '#DECA59' #'#CBC034' 
color_AS = '#6CA6F0'
color_blend = "blend:#6CA6F0,#702D96"
colorpalette_noAS = "blend:#DECA59,#DE6437"
sns.set_palette(color_blend, n_colors=3) # sns.set_palette("tab10")

def load_amsterdam_orientation_discrimination(filename, figs_dir):

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
    df_meta = df_meta_all.loc[(df_meta_all['event_type']=='iti'),:].reset_index(drop=True)
    df_meta['subject_id'] = subj
    df_meta['block_id'] = ses

    # remove doubles:
    df_meta = df_meta.iloc[np.where(df_meta['trial_nr'].diff()!=0)[0]]

    # add timestamps:
    df_meta['time_phase_1'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-1'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000
    df_meta['time_phase_2'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-2'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000
    df_meta['time_phase_10'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-10'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000
    df_meta['time_phase_11'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-11'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000
    df_meta['rt'] = df_meta['time_phase_11'] - df_meta['time_phase_10']

    # datetime
    timestamps = os.path.basename(filename).split('_')[2:]
    timestamps[-1] = timestamps[-1].split('.')[0]
    df_meta['start_time'] = datetime.datetime(*[int(t) for t in timestamps])
    df_meta['morning'] = df_meta['start_time'].dt.time <= datetime.time(13,0,0)
    
    # check if not too many omissions:
    if df_meta.shape[0]<85:
        print(subj)
        print(ses)
        return pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])

    # preprocess pupil data:
    fs = int(1/samples['time'].diff().median()*1000)
    params = {'fs':fs, 'lp':10, 'hp':0.01, 'order':3}
    df = preprocess_pupil.preprocess_pupil(samples=samples, events=events, params=params)
    df['time'] = df['time'] / 1000

    # make epochs:
    pupil_measure = 'pupil_int_lp_clean_psc'
    columns = ['subject_id', 'block_id', 'trial_nr', 'condition']

    epochs = utils.make_epochs(df=df, df_meta=df_meta, locking='time_phase_2', start=-2, dur=8, measure=pupil_measure, fs=fs, 
                    baseline=False, b_start=-1, b_dur=1)
    epochs[columns] = df_meta[columns]
    epochs_p_stim = epochs.set_index(columns)

    epochs = utils.make_epochs(df=df, df_meta=df_meta, locking='time_phase_11', start=-4, dur=8, measure=pupil_measure, fs=fs, 
                        baseline=False, b_start=-1, b_dur=1)
    epochs[columns] = df_meta[columns]
    epochs_p_resp = epochs.set_index(columns)

    # compute blinks and sacs:
    df_meta['blinks_ba_dec'] = [df.loc[(df['time']>(df_meta['time_phase_1'].iloc[i]-0))&(df['time']<df_meta['time_phase_10'].iloc[i]), 'is_blink_new'].mean() 
                for i in range(df_meta.shape[0])]
    df_meta['sacs_ba_dec'] = [df.loc[(df['time']>(df_meta['time_phase_1'].iloc[i]-0))&(df['time']<df_meta['time_phase_10'].iloc[i]), 'is_sac_eyelink'].mean() 
                for i in range(df_meta.shape[0])]
    df_meta['blinks_dec'] = [df.loc[(df['time']>(df_meta['time_phase_2'].iloc[i]-0))&(df['time']<df_meta['time_phase_10'].iloc[i]), 'is_blink_new'].mean() 
                for i in range(df_meta.shape[0])]
    df_meta['sacs_dec'] = [df.loc[(df['time']>(df_meta['time_phase_2'].iloc[i]-0))&(df['time']<df_meta['time_phase_10'].iloc[i]), 'is_sac_eyelink'].mean() 
                for i in range(df_meta.shape[0])]

    # plot result:
    plot = 1
    if plot:
        try:
            fig = utils.plot_blockwise_pupil(df_meta, df, events, pupil_measure)
            fig.savefig(os.path.join(figs_dir, 'preprocess', 'amsterdam_orientation_discrimination', '{}_{}.pdf'.format(subj, ses)))
        except:
            pass
    
    # downsample:
    epochs_p_stim = epochs_p_stim.iloc[:,::10]
    epochs_p_resp = epochs_p_resp.iloc[:,::10]

    # return:
    plt.close('all')
    return df_meta, epochs_p_stim, epochs_p_resp


# Plot 2: normal pupil time course
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
        plt.axvspan(-0.5,0.5, color='grey', alpha=0.15)
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

    sns.despine(trim=True)
    plt.tight_layout()

    # for (subj), m in means.groupby(['subject_id']):
    #     means.loc[m.index[0],:] = means.loc[m.index[0],:] - means.loc[m.index[3],:]
    #     means.loc[m.index[1],:] = means.loc[m.index[1],:] - means.loc[m.index[3],:]
    #     means.loc[m.index[2],:] = means.loc[m.index[2],:] - means.loc[m.index[3],:]

    return fig

# Plot 3: differential pupil responses
def plot_pupil_responses_amsterdam_orientation(epochs):

    means = epochs.groupby(['subject_id', 'condition']).mean()
    for (subj), m in means.groupby(['subject_id']):
        means.loc[m.index[0],:] = means.loc[m.index[0],:] - means.loc[m.index[3],:]
        means.loc[m.index[1],:] = means.loc[m.index[1],:] - means.loc[m.index[3],:]
        means.loc[m.index[2],:] = means.loc[m.index[2],:] - means.loc[m.index[3],:]
    sems = means.groupby(['condition']).sem()
    means = means.groupby(['condition']).mean()

    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(111)
    x = np.array(means.columns, dtype=float)
    plt.axvline(0, color='k', ls='--', linewidth=0.5)
    # plt.axhline(0, color='black', lw=0.5)
    for i in range(3):
        plt.fill_between(x, means.iloc[i]-sems.iloc[i], 
                            means.iloc[i]+sems.iloc[i], 
                            color=sns.color_palette(color_blend,3)[i], alpha=0.2)
        plt.plot(x, means.iloc[i], color=sns.color_palette(color_blend,3)[i], label=means.iloc[i].name)
    plt.xlabel('Time from 1st stimulus (s)')
    plt.ylabel('Δ Pupil response (% change)')
    # plt.legend()
    # plt.axvspan(0,2, color='grey', alpha=0.2)
    sns.despine(trim=True)
    plt.tight_layout()

    return fig