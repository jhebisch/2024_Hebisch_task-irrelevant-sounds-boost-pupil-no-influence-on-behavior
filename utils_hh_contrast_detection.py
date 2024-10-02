'''
This script contains functions needed in the preprocessing (0_preprocess_hh_contrast_detection) 
and analysis scripts (2_hh_contrast_detection) of experiment 2 
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
from pingouin import bayesfactor_ttest
from pingouin import ttest
import math
import statistics

import utils

# set plotting style
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
color_noAS = '#DECA59' 
color_AS = '#6CA6F0'
color_blend = "blend:#6CA6F0,#702D96"
colorpalette_noAS = "blend:#DECA59,#DE6437"
sns.set_palette(color_blend, n_colors=9)

def load_data(filename):
    
    '''
    Function for loading behavioral and eye data, preprocessing and epoching.

    Input
        - filename of a block's edf file

    Output
        - df_meta = behavioral data frame
        - epochs... = epoched eye data (one trial one row)
    '''
    # additional imports
    from pyedfread import edf
    import preprocess_pupil

    #extract subject and session
    subj = os.path.basename(filename).split('_')[0]
    ses = os.path.basename(filename).split('_')[1]
    print(subj)
    print(ses)

    # load pupil data and meta data: 
    try:
        samples, events, messages = edf.pread(filename, trial_marker=b'')
        df_meta_all = pd.read_csv(filename[:-4]+'_events.tsv', sep='\t') 
    except Exception as e:
        print(subj, ses, 'loading data failed')
        print(e)
        return pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])

    # clean up samples dataframe
    samples = samples.loc[samples['time']!=0,:].reset_index(drop=True) #subset rows with time != 0 
    if subj == '99' and ses == '2':
        samples = samples.rename({'pa_right': 'pupil'}, axis=1) # rename pupil area left to just pupil
    elif os.getcwd() == 'C:\\Users\\Josefine\\Documents\\Promotion\\arousal_memory_experiments\\data\\pilot2_contrast_detection':
        samples = samples.rename({'pa_right': 'pupil'}, axis=1)
    else:
        samples = samples.rename({'pa_left': 'pupil'}, axis=1) # rename pupil area left to just pupil
    messages = messages.reset_index(drop=True)

    # clean meta data and add useful columns 
    ## subset behav data to only response rows and add subject and block id:
    df_meta = df_meta_all.loc[(df_meta_all['phase']==2)&(df_meta_all['event_type']=='response')].reset_index(drop=True) # only for tasks with a response
    df_meta = df_meta.loc[(df_meta['response']=='y')|(df_meta['response']=='m'),:].reset_index(drop=True) # only for tasks with a response
    #df_meta= df_meta_all.loc[(df_meta_all['phase']==1)].reset_index() # we are only interested in the decision phase meta data + do I also need to include drop=True or no (because I'm keeping all trials)?
    df_meta['subject_id'] = subj
    df_meta['block_id'] = ses
    #df_meta = df_meta.iloc[3:-1] #cut off first and last rows which don't have the same amount of timephases

    ## remove the unnecessary cases where there's still 2 lines per trial:
    df_meta = df_meta.iloc[np.where(df_meta['trial_nr'].diff()!=0)[0]]
    
    ## remove 10 s break trials:
    df_meta = df_meta.loc[(df_meta['trial_nr']!=0)&(df_meta['trial_nr']!=1)&(df_meta['trial_nr']!=102)&(df_meta['trial_nr']!=203)&(df_meta['trial_nr']!=304)&(df_meta['trial_nr']!=405)] #cut off baseline trials
    df_meta=df_meta.reset_index(drop=True)

    ## add timestamps from eyetracker messages (for the start of each phase (0=ITI, 1=baseline, 2=decision phase), for the response onset) calculate rt: 
    df_meta['time_phase_0'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-0'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000 
    df_meta['time_phase_1'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-1'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000
    df_meta['time_phase_2'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-2'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000 
    ### make variable for actual auditory stimulus onset, list is a helper
    df_meta['actual_stim_start2']= np.array([messages.loc[messages['trialid ']=='post_sound_trial_{}'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000 #stays sequential and rows are not (?) in line with trial nr in df meta
    list_actual_stim_start=[]
    for i in range(0,len(df_meta['actual_stim_start2'])):
        try:
            k=df_meta['actual_stim_start2'][i].values[0]
            list_actual_stim_start.append(k)
        except:
            list_actual_stim_start.append(np.NaN)
    df_meta['actual_stim_start']=list_actual_stim_start
    df_meta['actual_soa'] = df_meta['actual_stim_start'] - df_meta['time_phase_2']
    ### make list containing all trialid messages that belong to a valid response (that was given in phase 2):
    df_resp=pd.DataFrame(columns=['trial_nr','trial_id'])
    for i in df_meta['trial_nr']: 
        value='start_type-response_trial-'+str(i)+'_' #create string that belongs to trial
        for j in messages['trialid ']:
            if value in j and 'phase-2' in j and ('key-m' in j or 'key-y' in j): #check whether response was y or m and given in phase 2
                row={"trial_nr":i,"trial_id":j}
                df_resp = pd.concat([df_resp, pd.DataFrame([row])])
    df_resp = df_resp.iloc[np.where(df_resp['trial_nr'].diff()!=0)[0]] # remove the 2nd of double-responses
    df_meta['time_response'] = np.array([messages.loc[messages['trialid ']=='{}'.format(i), 'trialid_time']for i in df_resp['trial_id']]).ravel()/1000
    df_meta['rt'] = df_meta['time_response'] - df_meta['time_phase_2']
    df_meta['stim_start'] = df_meta['time_phase_2'] + df_meta['soa']
    #df_meta.loc[(df_meta.soa > df_meta.rt), 'condition'] = 'no_as' #label trials in which the AS wasn't played because the subj reacted beforehand

    # datetime
    timestamps = os.path.basename(filename).split('_')[2:]
    timestamps[-1] = timestamps[-1].split('.')[0]
    df_meta['start_time'] = datetime.datetime(*[int(t) for t in timestamps])
    df_meta['morning'] = df_meta['start_time'].dt.time <= datetime.time(13,0,0)
    
    # add bin number of soa to df_meta #TODO drop? and move to different place(as I'm not always using the same binning)?
    df_meta['soa_bin']=np.NaN
    if os.getcwd() == 'C:\\Users\\Josefine\\Documents\\Promotion\\arousal_memory_experiments\\data\\pilot2_contrast_detection':
        bins=[-3.043, -2.65, -2.3, -1.95, -1.6, -1.25, -0.9, -0.55, -0.2, 0.16] # added a little buffer for as that was played a little too early or a little too late #[-3, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6]
        bin_names=[1,2,3,4,5,6,7,8,9]  #[1,2,3,4,5,6,7,8,9,10,11,12]
    else:
        bins=[-3.043, -2.65, -2.3, -1.95, -1.6, -1.25, -0.9, -0.55, -0.2, 0.15, 0.51] # added a little buffer for as that was played a little too early or a little too late #[-3, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6]
        bin_names=[1,2,3,4,5,6,7,8,9,10]  #[1,2,3,4,5,6,7,8,9,10,11,12]
    df_meta['soa_bin']= pd.cut(df_meta['actual_soa'], bins, labels=bin_names)
    df_meta['soa_bin']=df_meta['soa_bin'].astype('float').fillna(0).astype('int') #make nans zeros for simplicity of depicting no stim categories in the end
    df_meta.loc[(df_meta['condition']=='boost') & (df_meta['actual_soa'].isnull()),'soa_bin']=np.NaN
    
    ## count occurences of different bins
    counts = df_meta.groupby('soa_bin')['soa_bin'].count()
    print(subj, ses, 'Bin counts:', counts)

    ## make variables indicating hit(=1), miss(=2), fa(=3) or cr(=4) and correctness
    df_meta['sdt'] = np.nan
    df_meta['correct'] = np.nan
    if int(subj)<=20:
        df_meta.loc[(df_meta['stimulus']=='present') & (df_meta['response']=='m'), 'sdt'] = 1
        df_meta.loc[(df_meta['stimulus']=='present') & (df_meta['response']=='y'), 'sdt'] = 2
        df_meta.loc[(df_meta['stimulus']=='absent') & (df_meta['response']=='m'), 'sdt'] = 3
        df_meta.loc[(df_meta['stimulus']=='absent') & (df_meta['response']=='y'), 'sdt'] = 4
    else:
        df_meta.loc[(df_meta['stimulus']=='present') & (df_meta['response']=='y'), 'sdt'] = 1
        df_meta.loc[(df_meta['stimulus']=='present') & (df_meta['response']=='m'), 'sdt'] = 2
        df_meta.loc[(df_meta['stimulus']=='absent') & (df_meta['response']=='y'), 'sdt'] = 3
        df_meta.loc[(df_meta['stimulus']=='absent') & (df_meta['response']=='m'), 'sdt'] = 4
    df_meta.loc[(df_meta['sdt']==1) | (df_meta['sdt']==4), 'correct'] = 1
    df_meta.loc[(df_meta['sdt']==2) | (df_meta['sdt']==3), 'correct'] = 0
    
    # work on pupil data --> creates df, includes preprocessing (refering to preprocess pupil script)
    ## preprocess pupil data (refers to preprocess_pupil script):
    ### make a dataframe containing the black and white screen onset times for range normalisation
    df_bw = df_meta_all.loc[(df_meta_all['event_type']=='bw1')].reset_index(drop=True) # only for tasks with a response
    df_bw['subject_id'] = subj
    df_bw['block_id'] = ses
    ### define sampling frequency and parameters for filters
    fs = int(1/samples['time'].diff().median()*1000) 
    params = {'fs':fs, 'lp':10, 'hp':0.01, 'order':3}
    ### apply preprocessing
    df = preprocess_pupil.preprocess_pupil(samples=samples, events=events, params=params, df_bw=df_bw)
    df['time'] = df['time'] / 1000

    # get baseline epochs
    df_meta_bl = df_meta_all.loc[(df_meta_all['phase']==0) & (df_meta_all['event_type']=='bl')].reset_index(drop=True)
    df_meta_bl['subject_id'] = subj
    df_meta_bl['block_id'] = ses
    df_meta_bl['time_phase_0_bl'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-0'.format(i), 'trialid_time'] for i in df_meta_bl['trial_nr']]).ravel() / 1000
    df_meta_bl['time_phase_1'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-1'.format(i), 'trialid_time'] for i in df_meta_bl['trial_nr']]).ravel() / 1000
    epochs_p_bl10 = create_bl_epochs(df_meta=df_meta_bl, df=df, fs=fs)
    epochs_cp_bl10 = create_bl_epochs(df_meta=df_meta_bl, df=df, fs=fs, pupil_measure='pupil_int_lp_clean_cp')

    # create epochs containing pupil size and epochs containing blinks
    (df_meta, epochs_p_stim, epochs_p_resp, epochs_p_dphase) = create_epochs(df_meta, df, fs) #refers to create_epochs function that is also defined in this script
    ## using the custom pupil range (from black and white screen pupil response)
    (_, epochs_cp_stim, epochs_cp_resp, epochs_cp_dphase) = create_epochs(df_meta, df, fs, pupil_measure='pupil_int_lp_clean_cp') 

    return df_meta, epochs_p_stim, epochs_p_resp, epochs_p_dphase, epochs_p_bl10, epochs_cp_bl10, epochs_cp_stim, epochs_cp_resp, epochs_cp_dphase

def create_epochs(df_meta, df, fs, pupil_measure = 'pupil_int_lp_clean_psc'): #uses utils.py script 
    
    '''
    Input
        - df_meta: meta data
        - df: dataframe containing preprocessed eye data
        - fs: sampling frequency
        - pupil_measure: defines which pupil measure from preprocessing is used (e.g. which method of normalisation; defaults to interpolated lowpassfiltered cleaned for blink response and median normalised pupil size)

    Output
        - df_meta: meta data now including columns for amount of blinks and saccades
        - epochs_p_stim: epochs locked to task-irrelevant white noise stimulus
        - epochs_p_resp: epochs locked to button press
        - epochs_p_dphase: epochs locked to onset of decision phase
    '''

    import utils
    columns = ['subject_id', 'block_id', 'trial_nr', 'condition', 'actual_soa', 'soa_bin'] #it's called condition here

    # pupil size locked to the start of the task-irrelevant white noise stimulus
    epochs = utils.make_epochs(df=df, df_meta=df_meta, locking='actual_stim_start', start=-2, dur=7, measure=pupil_measure, fs=fs,   #[df_meta['condition']=='boost'] #'actual_stim_start'
                    baseline=False, b_start=-1, b_dur=1)
    epochs[columns] = df_meta[columns]
    epochs_p_stim = epochs.set_index(columns)

    # pupil size locked to the response
    epochs = utils.make_epochs(df=df, df_meta=df_meta, locking='time_response', start=-5, dur=7, measure=pupil_measure, fs=fs, 
                        baseline=False, b_start=-1, b_dur=1)
    
    epochs[columns] = df_meta[columns]
    epochs_p_resp = epochs.set_index(columns)

    # pupil size locked to decision phase onset:
    epochs = utils.make_epochs(df=df, df_meta=df_meta, locking='time_phase_2', start=-4, dur=7, measure=pupil_measure, fs=fs, 
                        baseline=False, b_start=-1, b_dur=1)
    
    epochs[columns] = df_meta[columns]
    epochs_p_dphase = epochs.set_index(columns) #for onset decision phase

    # add blinks and sacs to df_meta: copy from JW once he's synced his new version
    df_meta['blinks_ba_dec'] = [df.loc[(df['time']>(df_meta['time_phase_1'].iloc[i]-0))&(df['time']<df_meta['time_response'].iloc[i]), 'is_blink_new'].mean() 
                for i in range(df_meta.shape[0])]
    df_meta['sacs_ba_dec'] = [df.loc[(df['time']>(df_meta['time_phase_1'].iloc[i]-0))&(df['time']<df_meta['time_response'].iloc[i]), 'is_sac_eyelink'].mean() 
                for i in range(df_meta.shape[0])]
    df_meta['blinks_dec'] = [df.loc[(df['time']>(df_meta['time_phase_2'].iloc[i]-0))&(df['time']<df_meta['time_response'].iloc[i]), 'is_blink_new'].mean() 
                for i in range(df_meta.shape[0])]
    df_meta['sacs_dec'] = [df.loc[(df['time']>(df_meta['time_phase_2'].iloc[i]-0))&(df['time']<df_meta['time_response'].iloc[i]), 'is_sac_eyelink'].mean() 
                for i in range(df_meta.shape[0])]

    # downsample
    epochs_p_stim = epochs_p_stim.iloc[:,::10]
    epochs_p_resp = epochs_p_resp.iloc[:,::10]
    epochs_p_dphase = epochs_p_dphase.iloc[:,::10]

    return df_meta, epochs_p_stim, epochs_p_resp, epochs_p_dphase

def create_bl_epochs(df_meta, df, fs, pupil_measure='pupil_int_lp_clean_psc'): #uses utils.py script #TODO work here
    
    '''
        Input
        - df_meta: meta data
        - df: dataframe containing preprocessed eye data
        - fs: sampling frequency
        - pupil_measure: defines which pupil measure from preprocessing is used (e.g. which method of normalisation; defaults to interpolated lowpassfiltered cleaned for blink response and median normalised pupil size)

    Output
        - epochs_p_bl10: epochs locked to the onset of the 10 s breaks (5 per block)
    '''

    import utils

    # pupil size locked to the start of 10 s breaks
    epochs = utils.make_epochs(df=df, df_meta=df_meta, locking='time_phase_0_bl', start=2, dur=7, measure=pupil_measure, fs=fs,   #[df_meta['condition']=='boost'] #'actual_stim_start'
                    baseline=False, b_start=-1, b_dur=1)
    base_columns = ['subject_id', 'block_id', 'trial_nr']
    epochs[base_columns] = df_meta[base_columns]
    epochs_p_bl10 = epochs.set_index(base_columns)

    # downsample
    epochs_p_bl10 = epochs_p_bl10.iloc[:,::10]

    return epochs_p_bl10

def create_diff_epochs(epoch_dphase_in, lower=-0.96, dur=3.29, groupby=['subject_id']):

    # means = epoch_dphase_in.groupby(groupby).mean() 
    # means = means.loc[means.index.get_level_values(1)=='normal']
    epoch_diff_dphase = epoch_dphase_in.copy() 
    epoch_diff_dphase = epoch_diff_dphase.groupby(groupby).transform(lambda x: x - x.loc[x.index.get_level_values(3)=='normal'].mean())
    # epochs_p_dphase_d = epochs_p_dphase_d.loc[(epochs_p_dphase_b.index.get_level_values(0)==1)] - means.loc[(means.index.get_level_values(0)==1)]

    # create epochs from epochs_p_dphase_d that are locked to the respective actual soa
    start_inds = np.array(np.round((epoch_diff_dphase.index.get_level_values(4) + lower), 2))
    start_inds[start_inds < -4] = -4
    end_inds = np.array([round(i + dur, 2) for i in start_inds])
    epoch_diff_stim = []
    for s, e, i in zip(start_inds, end_inds, epoch_diff_dphase.index):
        if pd.isna(s):
            epoch = np.full(int((dur+0.01)*100), np.nan) #for not downsampled epochs it should be *1000
        else:
            epoch = np.array(epoch_diff_dphase.loc[i, s:e]) 
        epoch_diff_stim.append(epoch)
    epoch_diff_stim = pd.DataFrame(epoch_diff_stim)
    epoch_diff_stim.index = epoch_diff_dphase.index
    epoch_diff_stim.columns = np.arange(lower, lower+dur+0.01, 1/100).round(5)

    return epoch_diff_stim, epoch_diff_dphase

def compute_accuracy(df_meta, trial_cutoff):
    '''
    Computes accuracy for behavioral dataframe using a trial cutoff.
    '''
    df_meta['correct_avg'] = df_meta.loc[(df_meta['trial_nr']>trial_cutoff)&(df_meta['condition']==0)&(df_meta['correct']!=-1), 'correct'].mean()
    return df_meta

def compute_sequential(df):
    '''
    Creates columns in behavioral dataframe that indicate whether the last/next
    trial had the same response or not.
    '''
    df['repetition_past'] = df['response']==df['response'].shift(1) #asks: is the last one the same
    df['repetition_future'] = df['response']==df['response'].shift(-1) #asks: is the next one the same
    df['choice_p'] = df['choice'].shift(1)
    df['stimulus_p'] = df['stimulus'].shift(1)
    return df

def sdt(df_meta):

    '''
    Input
        - df_meta: behavioral data frame
    
    Output
        - dataframe that shows the signial detection theory indices d' (sensitivity), c (criterion), c_abs (absolute criterion), hr (hitrate) and far (false alarm rate)
    '''
    n_hit = (df_meta['sdt']==1).sum()
    n_miss = (df_meta['sdt']==2).sum()
    n_fa = (df_meta['sdt']==3).sum()
    n_cr = (df_meta['sdt']==4).sum()

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
    c_abs = abs(c)

    return pd.DataFrame({'d':[d], 'c':[c], 'c_abs':[c_abs], 'hr':[hit_rate], 'far':[fa_rate]})

# for making column in df called 'pupil_r_diff' which is the pupil response to AS
def compute_pupil_response_diff(epoch_diff_stim, epochs_p_stim): 

    # epochs that go in are already baselined
    x = epoch_diff_stim.columns
    means = epoch_diff_stim.loc[:,(x>0)&(x<2)].mean(axis=1)
    means = pd.DataFrame(means)
    
    scalars = pd.DataFrame()
    scalars.index = epochs_p_stim.index
    result = pd.concat([scalars, means], axis=1)
    result.columns = ['pupil_r_diff', *result.columns[1:]]

    return result

def compute_pupil_scalars(epochs, df, condition, dur=2.5, lower=-0.5): 

    epochs['fake_soa'] = df['fake_soa'].values
    epochs = epochs.set_index(['fake_soa'], append=True)
    scalars = np.repeat(0, epochs.shape[0])
    scalars = pd.DataFrame(scalars)
    scalars.index = epochs.index

    # make task-irrelevant stimulus-locked epochs
    epochs = epochs.query('condition == "{}"'.format(condition))
    if condition == "normal":
        start_inds = np.array(np.round((epochs.index.get_level_values('fake_soa')+lower), 2))
    elif condition == "boost":
        start_inds = np.array(np.round((epochs.index.get_level_values('actual_soa')+lower), 2))
    end_inds = np.array([round(i + dur, 2) for i in start_inds])
    epochs2 = []
    for s, e, i in zip(start_inds, end_inds, epochs.index):
        if pd.isna(s):
            epoch = np.full(int((dur+0.01)*100), np.nan) #for not downsampled epochs it should be *1000
        else:
            epoch = np.array(epochs.loc[i, s:e]) 
        epochs2.append(epoch)
    epochs2 = pd.DataFrame(epochs2)
    epochs2.index = epochs.index
    epochs2.columns = np.arange(lower, lower+dur+0.01, 1/100).round(5)
    # baseline:
    x = epochs2.columns
    epochs2 = epochs2 - np.atleast_2d(epochs2.loc[:,(x>-0.5)&(x<0)].mean(axis=1).values).T

    # get mean value over each epoch starting from task-irrelevant stimulus onset (0)
    scalars.loc[scalars.index.get_level_values('condition')==condition] = epochs2.loc[:,(x>0)].mean(axis=1)
    scalars = scalars.reset_index().iloc[:,-1]

    return scalars

# for heatmap across trials (add-on, no figure) calculate the mean of variables by trial num
def vars_across_trials(df, windowsize, stepsize, n_jobs, n_boot):
    means = []
    sems = []
    means.append(utils.compute_results(df.loc[df['condition']==0,:], groupby=['trial_nr']))
    sem_ = pd.concat(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(utils.compute_results)
                                                (df.loc[df['condition']==0,:].sample(frac=1, replace=True), ['trial_nr'], i)
                                                for i in range(n_boot)))
    sems.append((sem_.groupby(['trial_nr']).quantile(0.84) - sem_.groupby(['trial_nr']).quantile(0.16))/2)

    iter = np.arange(0,int(((3500-(windowsize * 1000))/(stepsize * 1000))+1),1) #526 --> was that correct? shouldn't it have been 525?
    start = float(-3)
    end = float(-3+windowsize)
    starts = []

    for i in iter:
        start = round(start, 3)
        end = round(end, 3)
        df['in_bin'] = 0
        df.loc[(df['actual_soa']>= start) & (df['actual_soa']<=end),'in_bin'] = 1
        means.append(utils.compute_results(df.loc[df['in_bin']==1,:], groupby=['trial_nr']))
        sem_ = pd.concat(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(utils.compute_results)
                                                    (df.loc[df['in_bin']==1,:].sample(frac=1, replace=True), ['trial_nr'], i)
                                                    for i in range(n_boot)))
        sems.append((sem_.groupby(['trial_nr']).quantile(0.84) - sem_.groupby(['trial_nr']).quantile(0.16))/2)
        starts.append(start)
        start += stepsize
        end += stepsize

    for i in range(len(sems)):
        sems[i] = sems[i].reset_index()

    return means, sems

# -------------------------------------------------
# Figures and analyses functions
# -------------------------------------------------

# Figure 2
# -------------------------------------------------
# Fig 2A normal pupil time courses by tpr bin
def plot_normal_pupil(epochs_p_dphase, locking, draw_box=False, shade=False):
    
    means = epochs_p_dphase.groupby(['subject_id', 'condition', 'tpr_bin']).mean().groupby(['condition', 'tpr_bin']).mean()
    sems = epochs_p_dphase.groupby(['subject_id', 'condition', 'tpr_bin']).mean().groupby(['condition', 'tpr_bin']).sem()
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

    sns.despine(trim=True)
    plt.tight_layout()
    return fig

# Fig 2B plot for pupil response to task-irrelevant sound time course by SOA bin
def plot_pupil_sliding_bin(epochs_p_dphase, epochs_diff_dphase, windowsize, stepsize, groupby=['subject_id'], bin_nrs=10, ylim=(-2,4)):
    # if bin_nrs==10:
    #     binning=['(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'] 
    
    means = epochs_p_dphase.groupby(['subject_id', 'condition']).mean().groupby(['condition']).mean()
    sems = epochs_p_dphase.groupby(['subject_id', 'condition']).mean().groupby(['condition']).sem()
    
    fig = plt.figure(figsize=(2,2))
    
    ax = fig.add_subplot(111)
    plt.axvline(0, color='k', ls='--', linewidth=0.5)
    plt.axvspan(-0.15, 1.85, color='grey', alpha=0.15)
    patch = []
    iter = np.arange(0,int(((3500-(windowsize * 1000))/(stepsize * 1000))+1),1) #526 --> was that correct? shouldn't it have been 525?
    start = float(-3)
    end = float(-3+windowsize)
    starts = []
    ends = []
    for i in iter:
        start = round(start, 3)
        end = round(end, 3)
        means = epochs_diff_dphase.loc[(epochs_diff_dphase.index.get_level_values(4)>=start) & (epochs_diff_dphase.index.get_level_values(4)<=end)].groupby(groupby).mean().mean()
        sems = epochs_diff_dphase.loc[(epochs_diff_dphase.index.get_level_values(4)>=start) & (epochs_diff_dphase.index.get_level_values(4)<=end)].groupby(groupby).mean().sem()
        x = np.array(means.index, dtype=float)
        plt.fill_between(x, means-sems, means+sems, color=sns.color_palette(color_blend,(len(iter)+1))[i+1], alpha=0.2)
        plt.plot(x, means, color=sns.color_palette(color_blend, (len(iter)+1))[i+1], ls='-') # , lw=0.8)
        # ax.set_yticks(np.arange()) #TODO
        patch.append(mpatches.Patch(color=sns.color_palette(color_blend, (len(iter)+1))[i+1], label= '({})-({})'.format(start,end)))
        start += stepsize
        end += stepsize

    # fig.legend(title= 'AS SOA bin (s)', title_fontsize = 7, handles=patch, loc='upper left', bbox_to_anchor=(1, 1))    
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Time from decision interval onset (s)')
    plt.ylabel('Δ Pupil response (% change)') #(% change)')
             
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

# Fig 2C plot for pupil resp to task-irrelevant sound by participant (scatter plot)
def plot_p_stats_sliding_bin(epoch_diff_stim, windowsize, stepsize, maxi=2, groupby=['subject_id'], ylim=(-2,4)):
    
    iter = np.arange(0,int(((3500-(windowsize * 1000))/(stepsize * 1000))+1),1) 
    if (len(iter) > 1):
        fig = plt.figure(figsize=(2,2))
    else:
        fig = plt.figure(figsize=(1,2))
    ax = fig.add_subplot(111)
    plt.axhline(0, color='k', linewidth=0.5) #ls='--', 
    # sns.stripplot(x=0, y=np.zeros(len(epoch_diff_stim.index.get_level_values(0).unique())), color=color_noAS, linewidth=0.2)
    # patch = []
    start = float(-3)
    end = float(-3+windowsize)
    starts = []
    t_values = []
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    y = np.array(epoch_diff_stim.columns, dtype=float)
    df_means = pd.DataFrame()
    for i in iter:
        start = round(start, 3)
        end = round(end, 3)
        means = epoch_diff_stim.loc[(epoch_diff_stim.index.get_level_values(4)>=start) & (epoch_diff_stim.index.get_level_values(4)<=end),(y>0)&(y<maxi)].groupby(groupby, sort=False).mean().mean(axis=1) #.max(axis=1)
        sns.stripplot(x=i, y=means.values, color=sns.color_palette(color_blend, (len(iter)+1))[i], linewidth=0.2, alpha=0.7, edgecolor=sns.color_palette(color_blend, (len(iter)+1))[i]) 
        ax.hlines(y=means.values.mean(), xmin=(i)-0.4, xmax=(i)+0.4, zorder=10, colors='k')
        # print(means.values.mean())
        t, p = stats.ttest_1samp(means, 0)
        p = p * len(iter)
        bf_value = bayesfactor_ttest(t, len(means), paired=True)
        print('BF window start', start , '{0: .3g},'.format(round(bf_value,3)))
        # plt.text(x=i, y=0.9, s=str(round(p,3)), size=6, rotation=45, transform=trans)
        # plt.text(x=i, y=1, s='$BF_{{{0:}}}={1: .3g}$'.format('10',round(bf_value,3)), size=5, rotation=90, transform=trans)
        # patch.append(mpatches.Patch(color=sns.color_palette(color_blend, (len(iter)+1))[i+1], label= '({})-({})'.format(start,end)))
        df_means[str(i)] = means
        starts.append(start)
        t_values.append(t)
        start += stepsize
        end += stepsize

    df_means[str(i+1)] = 0 # comment this line to only compare windows (and not normal condition)
    df_means = pd.melt(df_means.reset_index(), id_vars='subject_id')
    m = AnovaRM(data=df_means, depvar='value', subject='subject_id', within=['variable'])
    print(m.fit())

    # df_means = pd.melt(df_means.reset_index(), id_vars='subject_id')
    # m = AnovaRM(data=df_means, depvar='value', subject='subject_id', within=['variable'])
    # print(m.fit())
    # fig.legend(title= 'AS SOA bin (s)', title_fontsize = 7, handles=patch, loc='upper left', bbox_to_anchor=(1, 1))    
    if ylim is not None:
        plt.ylim(ylim)
    if len(iter)>1:
        centers = [str(round(i + (windowsize/2),3)) for i in starts]
        # centers.insert(0,'no stim')
        ax.set_xticklabels(centers, rotation=45)
        ax.set_xlabel('Center of SOA window (s)')
    else:
        # ax.set_xticklabels(['AS trials'])
        ax.get_xaxis().set_visible(False)

    plt.ylabel('Δ Pupil response (% change)')
             
    sns.despine(trim=True)
    plt.tight_layout()
    return fig, df_means, m, t_values

# Figure 3
# -------------------------------------------------
# Figure 3B comparison between pupil response amplitude and absolute criterion of task-evoked and task-irrelevant sound evoked effects
def plot_compare(df, groupby=['subject_id', 'condition'], m = 'c_abs'):
    
    df_res = utils.compute_results(df=df, groupby=groupby)
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(1,1,1)
    means = df_res.groupby(['condition']).mean().reset_index()
    sems = df_res.groupby(['condition']).sem().reset_index()
    plt.errorbar(x=means.loc[means['condition']==0,'pupil_base'], y=means.loc[means['condition']==0,m], xerr=sems.loc[means['condition']==0,'pupil_base'], yerr=sems.loc[means['condition']==0,m], fmt='o', color=color_noAS, alpha=0.7)
    plt.errorbar(x=means.loc[means['condition']==1,'pupil_r_sound'], y=means.loc[means['condition']==1,m], xerr=sems.loc[means['condition']==1,'pupil_r_sound'], yerr=sems.loc[means['condition']==1,m], fmt='o', color=color_AS, alpha=0.7)
    plt.xlabel('Δ Pupil response\n(% signal change)')
    if m == 'c_abs':
        plt.ylabel('| Criterion c |')
    plt.xlim(-1,2.5)
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

## Figure 3C task-irrelevant stimulus-evoked effects in behaviour (compute results with sliding window)
def plot_res_sliding_bin(df, var, windowsize, stepsize, groupby=['subject_id', 'in_bin'], ylim=(-2,4)): # careful: groupby has to be same as when creating df_sdt_bin!
    if (var == 'c') or (var == 'd') or (var == 'rt'):
        fig = plt.figure(figsize=(2,2.5)) #4,2))
    else:
        fig = plt.figure(figsize=(2,2))
    # gs = gridspec.GridSpec(1, 4)
    ax = fig.add_subplot() # gs[0, :1])
    means =[]
    sems = []
    df_res = utils.compute_results(df, groupby=['subject_id', 'condition']) #for leaving out first 100 trials: .loc[(df['trial_nr']>102)]
    mean = df_res.loc[(df_res['condition']==0),var].mean()
    sem = df_res.loc[(df_res['condition']==0),var].sem()
    # # means.append(mean)
    # # sems.append(sem)
    # sns.stripplot(x= 0, y=df_res.loc[(df_res['condition']==0),var], color=color_noAS, linewidth=0.2, ax=ax)
    # # plt.violinplot(df_res.loc[(df_res['condition']==0),var], color=color_noAS ,positions= [0])
    # ax.set_xticklabels(['no AS'])
    # ax.tick_params(axis='x', rotation=55)
    # ax.set_ylabel('{}'.format(var), fontsize=9)
    # plt.subplots_adjust(wspace=0.5)

    # ax = fig.add_subplot(gs[0, 1:])
    iter = np.arange(0,int(((3500-(windowsize * 1000))/(stepsize * 1000))+1),1) #526 --> was that correct? shouldn't it have been 525?
    start = float(-3)
    end = float(-3+windowsize)
    starts = []
    ends = []
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    df_diffs = pd.DataFrame()

    for i in iter:
        start = round(start, 3)
        end = round(end, 3)
        df['in_bin'] = 0
        df.loc[(df['actual_soa']>= start) & (df['actual_soa']<=end),'in_bin'] = 1
        df.loc[(df['condition']==0),'in_bin'] = 2
        df_res = utils.compute_results(df, groupby=groupby) #for leaving out first 100 trials: .loc[(df['trial_nr']>102)]
        var_bins = np.array(df_res.loc[(df_res['in_bin']==1),var])
        var_norm = np.array(df_res.loc[(df_res['in_bin']==2),var])
        diff = var_bins-var_norm
        sns.stripplot(x= (i), y=diff, color=sns.color_palette(color_blend, (len(iter)+1))[(i)], linewidth=0.2, ax=ax, alpha=0.7, edgecolor=sns.color_palette(color_blend,(len(iter)+1))[(i)])
        ax.hlines(y=diff.mean(), xmin=(i)-0.4, xmax=(i)+0.4, zorder=10, colors='k')
        t, p = stats.ttest_1samp(diff, 0)
        bf = bayesfactor_ttest(t, len(df['subject_id'].unique()), paired=True)
        print(var, start, bf)
        if (var == 'c') or (var == 'd') or (var == 'rt'):
            plt.text(x=i, y=1, s='$BF_{{{}}}={}$'.format('10',round(bf,3)), size=5, rotation=90, transform=trans)
        # plt.violinplot(df_res.loc[(df_res['in_bin']==1),var], color=sns.color_palette(color_blend, (len(iter)+1))[i],positions=[(i)])
        mean_bin = df_res.loc[(df_res['in_bin']==1),var].mean()-mean
        sem_bin = df_res.loc[(df_res['in_bin']==1),var].sem()-sem
        means.append(mean_bin)
        sems.append(sem_bin)
        starts.append(start)
        ends.append(end)
        df_diffs[str(i)] = diff
        start += stepsize
        end += stepsize

    means = np.array(means)
    sems = np.array(sems)
    # iter = np.append(iter, int(((3500-(windowsize * 1000))/(stepsize * 1000))+1))
    plt.fill_between(iter, means-sems, means+sems, color=sns.color_palette(color_blend, (len(iter)+1))[0], alpha=0.2)
    plt.plot(iter, means, color=sns.color_palette(color_blend, (len(iter)+1))[0], ls='-', lw=0.8)

    df_diffs = df_diffs.reset_index()
    df_diffs[str(i+1)] = 0
    df_diffs = pd.melt(df_diffs, id_vars='index')
    m = AnovaRM(data=df_diffs, depvar='value', subject='index', within=['variable'])
    print(m.fit())

    centers = [str(round(i + (windowsize/2),3)) for i in starts]
    # centers.insert(0,'no stim')
    ax.set_xticklabels(centers, rotation=90)
    ax.set_xlabel('Center of SOA window (s)')
    if var == 'c_abs':
        # ax.set_title('Bias AS - no AS', pad=35,fontweight='bold')
        ax.set_ylabel('Δ | Criterion c |')
    if var == 'c':
        # ax.set_title('Bias AS - no AS', pad=35,fontweight='bold')
        ax.set_ylabel('Δ Criterion c')
    if var == 'd':
        # ax.set_title('Performance AS - no AS', pad=35,fontweight='bold')
        ax.set_ylabel("Δ Sensitivity d'")
    if var == 'rt':
        # ax.set_title('Reaction time AS - no AS', pad=35,fontweight='bold')
        ax.set_ylabel('Δ Reaction time')
    if var == 'correct':
        # ax.set_title('Accuracy AS - no AS', pad=35,fontweight='bold')
        ax.set_ylabel('Δ Accuracy')

    sns.despine(trim=True)
    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()

    return fig, df_diffs, m

# Figure 5
# -------------------------------------------------
def baseline_interaction(m,n,y,z):
    fig = plt.figure(figsize=(2.5,2.3))
    ax = fig.add_subplot() #121)

    if m == 'pupil_r_diff':
        ## model for cond_SOA, baseline and interaction effects in AS pupil response or sdt metrics
        model = AnovaRM(data=n.loc[n['cond_SOA']!=0], depvar=m, subject='subject_id', within=['cond_SOA', 'pupil_b_bin'])
        res = model.fit()

        ## plots for cond_SOA, baseline and interaction effects in AS pupil response or sdt metrics
        # sns.pointplot(data=n.loc[n['cond_SOA']!=0], x='pupil_b_bin', y=m, hue='cond_SOA', palette = [sns.color_palette(color_blend,3)[0],sns.color_palette(color_blend,3)[1],sns.color_palette(color_blend,3)[2]]) 
        # sns.lineplot(data=n.loc[n['cond_SOA']!=0], x='pupil_b_bin', y=m, hue='cond_SOA', errorbar='se', palette = [sns.color_palette(color_blend,4)[0],sns.color_palette(color_blend,4)[1],sns.color_palette(color_blend,4)[2],sns.color_palette(color_blend,4)[3]]) 
        for i in [1,2,3,4]:
            means = n.loc[n['cond_SOA']==i].groupby(['pupil_b_bin']).mean()
            sems = n.loc[n['cond_SOA']==i].groupby(['pupil_b_bin']).sem()
            plt.errorbar(x=means['pupil_b'], y=means[m], xerr=sems['pupil_b'], yerr=sems[m], fmt='o', color=sns.color_palette(color_blend,4)[i-1], alpha=0.7)
        plt.ylabel('Δ Pupil response (% change)')
        coefs = ['SOA:           ', 'Pupil:          ', 'Interaction: ']
        for i in range(3):
            coef = coefs[i]
            f = res.anova_table["F Value"][i] #FIXME so far this is a 2-factor anova, here we're accessing the cond_SOA coefficient which is not complete and not the one of interest
            df1 = res.anova_table["Num DF"][i]
            df2 = res.anova_table["Den DF"][i]
            s = res.anova_table["Pr > F"][i]
            if s < 0.0001:
                txt = ' p < 0.001'
            else:
                txt = ' p = {}'.format(round(s,3))
            if s<0.05:
                plt.text(x=-21, y=y-(i*0.8), s='{}'.format(coef)+'$F_{{({},{})}}=$'.format(df1, df2)+str(round(f,3))+','+r'$\bf{{{}}}$'.format(txt), size=7, va='center', ha='left') # , c=color_AS)
            else:
                plt.text(x=-21, y=y-(i*0.8), s='{}'.format(coef)+'$F_{{({},{})}}=$'.format(df1, df2)+str(round(f,3))+','+txt, size=7, va='center', ha='left') # , c=color_AS)
    
    if m == 'c_abs_diff':
        ## model for cond_SOA, baseline and interaction effects in psychometric function
        model1 = AnovaRM(data=n, depvar=m, subject='subject_id', within=['cond_SOA', 'pupil_b_bin'])
        res1 = model1.fit()

        ##plot for cond_SOA, baseline and interaction effects in psychometric function
        # sns.lineplot(data=n, x='pupil_b_bin', y=m, hue='cond_SOA', errorbar='se', palette = [sns.color_palette(color_blend,4)[0],sns.color_palette(color_blend,4)[1],sns.color_palette(color_blend,4)[2],sns.color_palette(color_blend,4)[3]]) 
        for i in [1,2,3,4]:
            means = n.loc[n['cond_SOA']==i].groupby(['pupil_b_bin']).mean()
            sems = n.loc[n['cond_SOA']==i].groupby(['pupil_b_bin']).sem()
            plt.errorbar(x=means['pupil_b'], y=means[m], xerr=sems['pupil_b'], yerr=sems[m], fmt='o', color=sns.color_palette(color_blend,4)[i-1], alpha=0.7)
        coefs = ['SOA:           ', 'Pupil:          ', 'Interaction: ']
        for i in range(3):
            coef = coefs[i]
            f = res1.anova_table["F Value"][i] #FIXME so far this is a 2-factor anova, here we're accessing the cond_SOA coefficient which is not complete and not the one of interest
            df1 = res1.anova_table["Num DF"][i]
            df2 = res1.anova_table["Den DF"][i]
            s = res1.anova_table["Pr > F"][i]
            if s < 0.0001:
                txt = ' p < 0.001'
            else:
                txt = ' p = {}'.format(round(s,3))
            if s<0.05:
                plt.text(x=-21, y=y-0.02*i, s= '{}'.format(coef)+'$F_{{({},{})}}=$'.format(df1, df2)+str(round(f,3))+','+r'$\bf{{{}}}$'.format(txt), size=7, va='center', ha='left') # , c=sns.color_palette(color_blend,3)[i])
            else:
                plt.text(x=-21, y=y-0.02*i, s= '{}'.format(coef)+'$F_{{({},{})}}=$'.format(df1, df2)+str(round(f,3))+','+txt, size=7, va='center', ha='left') # , c=sns.color_palette(color_blend,3)[i])
        plt.ylabel('Δ | Criterion |')

    plt.ylim(z[0],z[1])
    plt.xlabel('Baseline pupil size (% w.r.t. mean)')
    # plt.xticks([0,1,2])
    plt.legend([], [], frameon=False)
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

# Figure S1
# -------------------------------------------------
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
# -------------------------------------------------
# Fig S2 A,B (baseline and tpr)
def plot_new_across_trials(means, sems, y, trial_cutoff=None):
    fig = plt.figure(figsize=(2,2))
    color_choices = [color_noAS,color_AS]
    plt.axvline(1, color='k', lw=0.5, ls='--')
    plt.axvline(102, color='k', lw=0.5, ls='--')
    plt.axvline(203, color='k', lw=0.5, ls='--')
    plt.axvline(304, color='k', lw=0.5, ls='--')
    plt.axvline(405, color='k', lw=0.5, ls='--')

    if trial_cutoff is not None:
        plt.axvline(trial_cutoff, color='k', lw=0.5)
    for i in range(len(means)):
        # means[i] = means[i].reset_index()
        x = np.array(means[i]['trial_nr'])
        # x = np.array(means[i].index)
        window = 10
        mean = means[i].rolling(window=window, center=True, min_periods=1).mean()
        sem = sems[i].rolling(window=window, center=True, min_periods=1).mean()
        plt.fill_between(x, mean.reset_index()[y]-sem.reset_index()[y], mean.reset_index()[y]+sem.reset_index()[y], alpha=0.2, color=color_choices[int(mean['condition'].mean())])
        plt.scatter(x, mean[y], color=color_choices[int(mean['condition'].mean())], s=2.7, marker='o', lw=0.05, alpha=0.7)
    plt.xlabel('Trial (#)')
    if (y=='pupil_b') | (y=='pupil_tpr_b'):
        plt.ylabel('Relative pupil size (%)')
    else:
        plt.ylabel('Pupil response (% change)')
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

# Fig S2 C pupil response to task-irrelevant sound means across trials
def plot_p_across_trials(means, sems, y, trial_cutoff=None):
    fig = plt.figure(figsize=(2,2))
    if trial_cutoff is not None:
        plt.axvline(trial_cutoff, color='k', lw=0.5)
    for i in range(len(means)):
        # means[i] = means[i].reset_index()
        # x = np.array(means[i]['trial_nr'])
        x = np.array(means[i].index)
        window = 10
        mean = means[i].rolling(window=window, center=True, min_periods=1).mean()
        sem = sems[i].rolling(window=window, center=True, min_periods=1).mean()
        plt.fill_between(x, mean[y]-sem[y], mean[y]+sem[y], alpha=0.2, color=color_AS)
        plt.scatter(x, mean[y], color=color_AS, s=2.7, marker='o', lw=0.05, alpha=0.7)
    plt.xlabel('Trial (#)')
    plt.title('Pupil response to AS', fontweight='bold')
    plt.ylabel('Pupil response (% change)')
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

# heatmap variable means across trials
def plot_across_trials(means, sems, y, trial_cutoff):
    fig = plt.figure(figsize=(3,2))
    plt.axvline(trial_cutoff, color='k', lw=0.5)
    mean = pd.DataFrame()
    x = np.array(means[0]['trial_nr'])
    for i in range(len(means)):
        window = 10
        mean[str(i)] = means[i].rolling(window=window, center=True, min_periods=1).mean()[y]
        # sem = sems[i].rolling(window=window, center=True, min_periods=1).mean()
        # plt.fill_between(x, mean[y]-sem[y], mean[y]+sem[y], alpha=0.2)
        # plt.plot(x, mean[y])
    mean = mean.transpose()
    sns.heatmap(mean, cbar_kws={'label': y})
    plt.xlabel('Trial (#)')
    plt.ylabel('AS SOA bin')
    sns.despine(trim=True)
    plt.tight_layout()
    return fig

# Figure S6
# -------------------------------------------------
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
            plt.ylabel('Δ | Criterion c |')
    sns.despine(trim=True)
    plt.tight_layout()
    return fig


# -------------------------------------------------
# currently not used
# -------------------------------------------------

# # function to read in all data that was saved in individual data frames
# def load_all_data(df_in, subjects, set_indexing=False, resp_soa_col=None):
#     files=['{}_{}.csv'.format(i,df_in) for i in subjects]
#     df_out = pd.concat(map(pd.read_csv, files), ignore_index=True)

#     #add resp_soa_bin (locked to response) and potentially change soa_bin (locked to decision phase onset; commented stuff)
#     # bins=[-3.043, -2.65, -2.3, -1.95, -1.6, -1.25, -0.9, -0.55, -0.2, 0.15, 0.51] # added a little buffer for as that was played a little too early or a little too late #[-3, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6]
#     # bin_names=[1,2,3,4,5,6,7,8,9,10]  #[1,2,3,4,5,6,7,8,9,10,11,12]
#     resp_bins=[-10, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0] # added a little buffer for as that was played a little too early or a little too late #[-3, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6]
#     resp_bin_names=[1,2,3,4,5,6,7,8,9,10,11]  #[1,2,3,4,5,6,7,8,9,10,11,12]

#     # df_out = binning(df=df_out, col_name='soa_bin', bins=bins, bin_names=bin_names, soa_col='actual_soa')
    
#     if resp_soa_col is None:
#         df_out['resp_soa'] = df_out['actual_stim_start'] - df_out['time_response']
#     if resp_soa_col is not None:
#         df_out['resp_soa'] = resp_soa_col.copy()

#     df_out = binning(df=df_out, col_name='resp_soa_bin', bins=resp_bins, bin_names=resp_bin_names, soa_col='resp_soa')

#     # set index on epochs
#     if set_indexing is True :
#         df_out=df_out.set_index(['subject_id', 'block_id', 'trial_nr', 'condition', 'actual_soa', 'resp_soa', 'soa_bin', 'resp_soa_bin'])

#     return df_out

# def binning(df, col_name, bins, bin_names, soa_col):
#     df[col_name] = np.NaN
#     df[col_name] = pd.cut(df[soa_col], bins, labels=bin_names)
#     df[col_name] = df[col_name].astype('float').fillna(0).astype('int') #make normal condition bin 0
#     df.loc[(df['condition']=='boost') & (df[soa_col].isnull()),col_name] = np.NaN #everything that doesn't fall into bins or normal cond should be a nan

#     return df

# # function for extracting peaks per participant from dictionary containing all peaks (see plot_pupil_responses_all), which_peak has option 'first' or 'max'
# def extract_peak(df, which_peak, time_lower, time_upper):
#     if which_peak == 'max':
#         peak = df.loc[(df['peak_ind']>=time_lower) & (df['peak_ind']<=time_upper), 'peak_val'].max()
#     if which_peak == 'first':
#         peaks = df.loc[(df['peak_ind']>=time_lower) & (df['peak_ind']<=time_upper), 'peak_val'].reset_index()
#         peak = peaks['peak_val'][0]

#     return peak #df contains a column for particpant number and a column for the chosen peak value

# # function for calculating the mean and max of AS pupil response of each subject
# def mean_p_resp(epoch_p, time_lower, time_upper, groupby=['block_id','trial_nr']):

#     dict_mean = {}
#     dict_max = {}
#     for sub_id, new_df in epoch_p.groupby('subject_id'):
#         means = new_df.groupby(groupby).mean().mean()
#         means.index = means.index.map(float)
#         means_mean = means.loc[(means.index >= time_lower) & (means.index <= time_upper)].mean()
#         dict_mean[str(sub_id)] = means_mean
#         means_max = means.loc[(means.index >= time_lower) & (means.index <= time_upper)].max()
#         dict_max[str(sub_id)] = means_max

#     return dict_mean, dict_max

# # function for getting max or mean of each epoch
# def max_mean_row(df_meta_whole, epoch_p, time_lower, time_upper):
#     max_mean = pd.DataFrame(epoch_p.loc[:,(epoch_p.columns >= time_lower) & (epoch_p.columns <= time_upper)].max(axis=1), columns=['p_trial_max'])
#     max_mean['p_trial_mean'] = epoch_p.loc[:,(epoch_p.columns >= time_lower) & (epoch_p.columns <= time_upper)].mean(axis=1)
#     max_mean['alt_index'] = max_mean.index.copy()
#     # means2 = max.groupby('subject_id').mean()
#     # where df_meta_whole alt index = max.index make column value max value:
#     df_meta_new = pd.merge(df_meta_whole, max_mean, on='alt_index', how='left')

#     return df_meta_new

# # function for calculating the correlation between a variable and change in that variable including checking correlation against regression to the mean
# def permutationTest_correlation(x1, x2, tail, nrand=10000):
#     """
#     calculates pearson r for correlation between x1 and the difference between x2 and x1 (x2-x1) and tests it with a permutation test

#     !!! attention: this code is meant for distribution/0-hypothesis lying completely on either side of 0, if 0-hypothesis is cor=0 see below (omit (*2)), if neither is the case: reconsider
    
#     this code is used here to get p-values corrected for regression to the mean, so we think our 0-hypothesis is cor is somewhat negative, and we wanna test whether our correlation is lower, so we use tail=-1
    
#     test whether 2 correlations are significantly different. For permuting single corr see randtest_corr2
#     function out = randtest_corr(a,b,tail,nrand, type)
#     tail = 0 (test A~=B), 1 (test A>B), -1 (test A<B)
#     type = 'Spearman' or 'Pearson'
#     """
#     rng = np.random.default_rng() #get a generator that randomly flips sign on conditions
#     data = np.vstack((x1,x2)).T #get variables of interest next to each other
#     corrrand = np.zeros(nrand) #np array to be filled with random correlations that arise when conditions are shuffled
#     truecorr = sp.stats.pearsonr(x1, x2-x1)[0]
#     for i in range(nrand):
#         data = rng.permuted(data, axis=1) #random walk generator permuted data
#         corrrand[i] = sp.stats.pearsonr(data[:,0], data[:,1]-data[:,0])[0]    
#     if tail == 0:
#         # p_value = (abs(corrrand) >= abs(truecorr)).mean()
#         p_value = (sum(abs(corrrand) >= abs(truecorr)) / float(nrand)) * 2 #* 2 added by me because i'm quite sure it's needed for 2-tailed testing when 0-hypothesis distribution lies completely in positive or negative values, if 0-hypothesis is r=0 -> leave out *2!
#     else:
#         p_value = sum(tail*(corrrand) >= tail*(truecorr)) / float(nrand)

#     return truecorr, corrrand, p_value

# # (for fig S3) sliding window graph of correlation between any sdt measure on normal trials and the difference towards AS trials within the respective window
# def sliding_window_cor_dev(df_meta_whole, measure, stepsize, windowsize, tail, sdt_groupby=['subject_id','in_bin']): #sliding step size = 5ms 
#     '''
#     Function that slides a window along boost condition AS SOAs (-3 to 0.5) in steps.
#     For each window we calculate that correlation between criterion (sdt) on no-stim (normal)
#     trials with the difference in between SOA window boost trials - no-stim trials

#     Each of these correlations is corrected for regression to the mean. Therefore, the tail enables us to 
#     decide between 1- and 2-tailed testing. We usually use 1-tailed.    
#     '''
    
#     true_corrs = []
#     p_values = []
#     corrands = []
#     # soas = np.arange(-3000,505,5)
#     # soas = pd.Series(soas)
#     # for window in soas.rolling(window=0.875):
#     iter = np.arange(0,int(((3500-(windowsize * 1000))/(stepsize * 1000))+1),1) #526 --> was that correct? shouldn't it have been 525?
#     start = float(-3)
#     end = float(-3+windowsize)
#     starts = []
#     ends = []
#     for i in iter: #iterate through windows
#         start = round(start, 3)
#         end = round(end, 3)
#         # create column that defines whether trial lies in window or not (or is in normal condition)
#         df_meta_whole['in_bin'] = 0
#         df_meta_whole.loc[(df_meta_whole['actual_soa']>= start) & (df_meta_whole['actual_soa']<=end),'in_bin'] = 1
#         df_meta_whole.loc[(df_meta_whole['condition']=='normal') | df_meta_whole['condition']==0,'in_bin'] = 2
#         # calculate sdt indices for the respective window and for normal condition
#         df_sdt = df_meta_whole.groupby(sdt_groupby).apply(sdt) #for leaving out first 100 trials: .loc[(df_meta_whole['trial_nr']>102)]
#         df_sdt.reset_index(inplace=True)
#         df_sdt = df_sdt.rename(columns = {'index':'in_bin'})
#         if sdt_groupby == ['subject_id', 'block_id', 'in_bin']:
#             df_sdt = df_sdt.groupby(['subject_id', 'in_bin']).mean()
#             df_sdt.reset_index(inplace=True)
#         # get correlation between sdt measure in window and sdt measure in normal condition and correct for regression towards the mean (by means of bootstrapping)
#         true_corr, corrand, p_value = permutationTest_correlation(df_sdt.loc[(df_sdt['in_bin']==2), measure].values, df_sdt.loc[(df_sdt['in_bin']==1), measure].values, tail=tail)
#         true_corrs.append(true_corr)
#         p_values.append(p_value)
#         corrands.append(list(corrand))
#         starts.append(start)
#         ends.append(end)
#         start += stepsize
#         end += stepsize

#     # get centers of windows and save stuff in dataframes
#     centers = [i + (windowsize/2) for i in starts]
#     df_cors_delta_c = pd.DataFrame(list(zip(centers, true_corrs, p_values)), columns=['centers', 'true_corrs','p_values'])
#     df_cors_delta_c['significance'] = 0
#     df_cors_delta_c.loc[(df_cors_delta_c['p_values']<0.05),'significance'] = 1

#     df_corrrands =  pd.DataFrame(corrands)

#     # cluster correction
#     rand_max_clusters, p_values_cluster, true_clusters = permutationTest_cluster(df_cors_delta_c, df_corrrands)
#     # fdr correction
#     p_true , adj_p = multitest.fdrcorrection(df_cors_delta_c['p_values'])
#     p_true = p_true.astype(int)

#     # plot results
#     fig, ax= plt.subplots(1,1,figsize=(3,2)) # figsize=(5,3.5))
#     ax.axvline(0, color='k', ls='dashed', lw=0.5)
#     # plt.plot(centers, true_corrs)
#     sns.lineplot(data=df_cors_delta_c,x=centers, y=true_corrs, palette=sns.color_palette("blend:#6CA6F0,#7931A3", as_cmap=True))
#     ax.set_xlabel('Center of AS SOA window (s)')
#     if measure == 'c':
#         ax.set_ylabel('Correlation Criterion & Δ Criterion') #(AS SOA window - no-AS condition)
#     else:
#         ax.set_ylabel('cor of {} & Δ {}'.format(measure, measure)) #(AS SOA window - no-AS condition)
#     # # shade area that is significant red, if it's still significant after cluster correction make it yellow
#     # start_span = list(df_cors_delta_c.iloc[np.where(df_cors_delta_c['significance'].diff()==1)]['centers'])
#     # end_span = list(df_cors_delta_c.iloc[np.where(df_cors_delta_c['significance'].diff()==-1)]['centers'])
#     # if (len(start_span)!=0) or (len(end_span)!=0):
#     #     if end_span[0] < start_span[0]:
#     #         start_span.insert(0,centers[0])
#     #     for i,j,k in zip(start_span, end_span, p_values_cluster):
#     #         if k <= 0.05:
#     #             ax.axvspan(i,(j-stepsize), alpha=0.1, color='yellow')
#     #         if k > 0.05:
#     #             ax.axvspan(i,(j-stepsize), alpha=0.07, color='red')
#     # shade area that is significant after fdr correction blue
#     start_span = list(df_cors_delta_c.iloc[np.where(np.diff(p_true)==1)]['centers'])
#     end_span = list(df_cors_delta_c.iloc[np.where(np.diff(p_true)==-1)]['centers'])   
#     if (len(start_span)!=0) or (len(end_span)!=0):
#         if end_span[0] < start_span[0]:
#             start_span.insert(0,centers[0])
#         for i,j in zip(start_span, end_span):
#             ax.axvspan((i+stepsize),(j), alpha=0.3)

#     sns.despine(trim=True)
#     plt.tight_layout()

#     return fig, df_cors_delta_c, df_corrrands, rand_max_clusters, p_values_cluster, true_clusters, adj_p

# def sliding_cor_rt(df_meta_whole, measure, stepsize, windowsize, tail, rt_groupby=['subject_id','in_bin']): #sliding step size = 5ms
#     '''
#     Function that slides a window of 875ms size along boost condition AS SOAs (-3 to 0.5) in steps of 5ms.
#     For each window we calculate that correlation between criterion (sdt) on no-stim (normal)
#     trials with the difference in between SOA window boost trials - no-stim trials

#     Each of these correlations is corrected for regression to the mean. Therefore, the tail enables us to 
#     decide between 1- and 2-tailed testing. We usually use 1-tailed.    
#     '''
    
#     true_corrs = []
#     p_values = []
#     corrands = []
#     # soas = np.arange(-3000,505,5)
#     # soas = pd.Series(soas)
#     # for window in soas.rolling(window=0.875):
#     iter = np.arange(0,int(((3500-(windowsize * 1000))/(stepsize * 1000))+1),1) #526 --> was that correct? shouldn't it have been 525?
#     start = float(-3)
#     end = float(-3+windowsize)
#     starts = []
#     ends = []
#     for i in iter:
#         start = round(start, 3)
#         end = round(end, 3)
#         df_meta_whole['in_bin'] = 0
#         df_meta_whole.loc[(df_meta_whole['actual_soa']>= start) & (df_meta_whole['actual_soa']<=end),'in_bin'] = 1
#         df_meta_whole.loc[(df_meta_whole['condition']=='normal'),'in_bin'] = 2
#         df_rt = df_meta_whole.groupby(rt_groupby).mean() #for leaving out first 100 trials: .loc[(df_meta_whole['trial_nr']>102)]
#         df_rt.reset_index(inplace=True)
#         df_rt = df_rt.rename(columns = {'index':'in_bin'})
#         if rt_groupby == ['subject_id', 'block_id', 'in_bin']:
#             df_rt = df_rt.groupby(['subject_id', 'in_bin']).mean()
#             df_rt.reset_index(inplace=True)

#         true_corr, corrand, p_value = permutationTest_correlation(df_rt.loc[(df_rt['in_bin']==2), measure].values, df_rt.loc[(df_rt['in_bin']==1), measure].values, tail=tail)
#         true_corrs.append(true_corr)
#         p_values.append(p_value)
#         corrands.append(list(corrand))
#         starts.append(start)
#         ends.append(end)
#         start += stepsize
#         end += stepsize

#     centers = [i + (windowsize/2) for i in starts]
#     df_cors_delta_rt = pd.DataFrame(list(zip(centers, true_corrs, p_values)), columns=['centers', 'true_corrs','p_values'])
#     df_cors_delta_rt['significance'] = 0
#     df_cors_delta_rt.loc[(df_cors_delta_rt['p_values']<0.05),'significance'] = 1

#     df_corrrands =  pd.DataFrame(corrands)

#     rand_max_clusters, p_values_cluster, true_clusters = permutationTest_cluster(df_cors_delta_rt, df_corrrands)

#     fig, ax= plt.subplots(1,1)
#     ax.axvline(0, color='grey', ls='dashed')
#     plt.plot(centers, true_corrs)
#     ax.set_xlabel('center of AS SOA window (windowsize={}s) in seconds relative to decision phase onset'.format(windowsize), fontsize=8)
#     ax.set_ylabel('cor of {} with Δ {} (AS SOA window - no-AS condition)'.format(measure, measure), fontsize=8)
#     start_span = list(df_cors_delta_rt.iloc[np.where(df_cors_delta_rt['significance'].diff()==1)]['centers'])
#     end_span = list(df_cors_delta_rt.iloc[np.where(df_cors_delta_rt['significance'].diff()==-1)]['centers'])
#     if (len(start_span)!=0) or (len(end_span)!=0):
#         if end_span[0] < start_span[0]:
#             start_span.insert(0,centers[0])
#         for i,j,k in zip(start_span, end_span, p_values_cluster):
#             if k <= 0.05:
#                 ax.axvspan(i,(j-stepsize), alpha=0.1)
#             if k > 0.05:
#                 ax.axvspan(i,(j-stepsize), alpha=0.07, color='red')
#     sns.despine(trim=True)
#     plt.tight_layout()

#     return fig, df_cors_delta_rt, df_corrrands, rand_max_clusters, p_values_cluster, true_clusters

# # cluster correction for p-values of sliding window correlation analysis
# def permutationTest_cluster(df_cors_delta_c, df_corrrands, nrand=10000):
#     '''
#     perform cluster correction for p-values of sliding window correlation analysis.
#     works with the permutations that were done for this analysis already to control for
#     regression to the mean.

#     in:
#      - df_cors_delta_c: dataframe that contains result of sliding window analysis
#             index = window index
#             centers = AS SOA window center
#             truecorrs = pearson r in observed data
#             p_values = p_value of each correlation corrected for regression to the mean
#             significance = 0:insignificanct, 1:significant
#      - df_corrrands: holds all pearson r values that were calculated in permutation test (randomizing boost vs. normal condition)
#      in each window in the correction for regression to the mean (permutationTest_correlation).
#      That way each column contains one permutation of the sliding window analysis

#     out:
#      - rand_max_clusters: list containing biggest sum of r-values of clusters of each random permutation
#      - p_values: for each true cluster's sum of r-values
#      - true_c: true clusters' sums of r-values
#     '''
    
#     true_c = [] # empty list to be filled with sum of r-values of true clusters
#     starts = list(df_cors_delta_c.loc[(df_cors_delta_c['significance'].diff()==1)].index)
#     ends = list(df_cors_delta_c.loc[(df_cors_delta_c['significance'].diff()==-1)].index)
#     if (len(starts)==0) & (len(ends)==0):
#         true_c = []
#     else:
#         if len(starts) == 0:
#             starts.insert(0,0)
#         elif ends[0] < starts[0]:
#             starts.insert(0,0)
#         for i,j in zip(starts, ends):
#             cluster = sum(df_cors_delta_c['true_corrs'][i:j]) # calculates sum of all true r values within each cluster
#             true_c.append(abs(cluster)) #take absolute sum
    
#     # if I wanted cluster size instead of r-value sum:
#     # true_c = np.diff(np.where(np.concatenate(([data[0]],
#     #                                  data[:-1] != data[1:],
#     #                                  [1])))[0])[::2] #calculates the given cluster sizes
    
#     # compute thresholds of pearson r for significance (alpha) for each window
#     thresholds = []
#     for i in range(len(df_corrrands)):
#         thresholds.append(np.percentile(df_corrrands.iloc[i,:], 5)) #take 5th percentile because we want to test whether our correlation is lower (more negative) than random regression to the mean
#     df_corrrands['thresholds'] = thresholds

#     rand_max_clusters = [] #list to be filled with max cluster sums of r-values of permutations of sliding window analysis
#     # look for significant random correlations in each iteration of the permutation of sliding window correlation 
#     # calculate clusters and their sum of r-values in each iteration and add max cluster to list
#     for i in range(nrand): 
#         df_corrrands['significance'] = 0
#         df_corrrands.loc[(df_corrrands[i]<df_corrrands['thresholds']), 'significance'] = 1 # check for each r-value of window (randomly assigned to condition) if it's smaller (more negative) than threshold
#         clusters = []
#         start = list(df_corrrands.loc[(df_corrrands['significance'].diff()==1)].index)
#         end = list(df_corrrands.loc[(df_corrrands['significance'].diff()==-1)].index)
#         if len(end) != 0:
#             if len(start) == 0:
#                 start.insert(0,0)
#             elif end[0] < start[0]:
#                 start.insert(0,0)
#         for k,l in zip(start, end):
#             cluster = sum(df_corrrands[i][k:l]) # calculates sum of all true r values within each cluster
#             clusters.append(abs(cluster)) #add absolute sum of each clusters r-values to list
#         try:
#             rand_max_clusters.append(max(clusters)) #each iteration take the max cluster and add to list of max_cluster that form the distribution to estimate p value with
#         except ValueError:
#             print(ValueError)
#             rand_max_clusters.append(0)
#     rand_max_clusters = np.array(rand_max_clusters)
#     # compare true clusters with random max clusters from above to calculate p-values for them
#     p_values = []
#     for j in true_c: 
#         p_values.append(sum(abs(rand_max_clusters) >= abs(j)) / float(len(rand_max_clusters))) # use percentile
    
#     return rand_max_clusters, p_values, true_c

# #-----------------------------------------------
# ####  PLOTS  ####
# #-----------------------------------------------


# # Fig S: history biases on bl and repetition probability
# def plot_history(df,m):
#     means = []
#     sems = []
#     means.append(df.loc[(df['condition']==1), m].mean())
#     means.append(df.loc[(df['condition'].shift(1)==1) & (df['condition']==0), m].mean())
#     means.append(df.loc[(df['condition'].shift(2)==1) & (df['condition']==0) & (df['condition'].shift(1)==0), m].mean())
#     means.append(df.loc[(df['condition'].shift(3)==1) & (df['condition']==0) & (df['condition'].shift(1)==0) & (df['condition'].shift(2)==0), m].mean())
#     means.append(df.loc[(df['condition'].shift(4)==1) & (df['condition']==0) & (df['condition'].shift(1)==0) & (df['condition'].shift(2)==0) & (df['condition'].shift(3)==0), m].mean())
#     means.append(df.loc[(df['condition'].shift(5)==1) & (df['condition']==0) & (df['condition'].shift(1)==0) & (df['condition'].shift(2)==0) & (df['condition'].shift(3)==0) & (df['condition'].shift(3)==0), m].mean())
#     sems.append(df.loc[(df['condition']==1), m].sem())
#     sems.append(df.loc[(df['condition'].shift(1)==1) & (df['condition']==0), m].sem())
#     sems.append(df.loc[(df['condition'].shift(2)==1) & (df['condition']==0) & (df['condition'].shift(1)==0), m].sem())
#     sems.append(df.loc[(df['condition'].shift(3)==1) & (df['condition']==0) & (df['condition'].shift(1)==0) & (df['condition'].shift(2)==0), m].sem())
#     sems.append(df.loc[(df['condition'].shift(4)==1) & (df['condition']==0) & (df['condition'].shift(1)==0) & (df['condition'].shift(2)==0) & (df['condition'].shift(3)==0), m].sem())
#     sems.append(df.loc[(df['condition'].shift(5)==1) & (df['condition']==0) & (df['condition'].shift(1)==0) & (df['condition'].shift(2)==0) & (df['condition'].shift(3)==0) & (df['condition'].shift(3)==0), m].sem())

#     fig = plt.figure(figsize=(2,2))
#     ax = fig.add_subplot(111)
#     x = np.array([0,1,2,3,4,5])
#     means = np.array(means)
#     sems = np.array(sems)
#     plt.fill_between(x, means-sems, 
#                         means+sems, alpha=0.2)
#     plt.plot(x, means)
#     plt.xlabel('AS x trials ago')
#     if (m == 'pupil_b') | (m == 'pupil_tpr_b'):
#         ax.set_title('Pupil baseline', fontweight='bold')
#         plt.ylabel('Pupil size (% signal)')
#     if m == 'repetition_past':
#         ax.set_title("Repetition of last trial's choice", fontweight='bold', x=0.4, y=1)
#         ax.set_ylabel('Repetition rate')
#     if m == 'repetition_future':
#         ax.set_title("Repetition of choice on next trial", fontweight='bold', x=0.3, y=1)
#         ax.set_ylabel('Repetition rate')
#     if (m == 'tpr') | (m == 'tpr_c'):
#         ax.set_title('Task evoked pupil response',fontweight='bold')
#         ax.set_ylabel("Pupil response (% change)")
#     ax.set_xticks([0,1,2,3,4,5])
#     # plt.legend()
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# # Fig S: sliding window correlation of delta pupil with c or delta c
# def plot_sliding_cors(df, epoch_diff_stim, measure, windowsize, stepsize, maxi=2, sdt_groupby=['subject_id', 'in_bin'], delta=False):

#     true_corrs = []
#     p_values = []
#     iter = np.arange(0,int(((3500-(windowsize * 1000))/(stepsize * 1000))+1),1) #526 --> was that correct? shouldn't it have been 525?
#     start = float(-3)
#     end = float(-3+windowsize)
#     starts = []
#     ends = []
#     y = np.array(epoch_diff_stim.columns, dtype=float)
#     for i in iter: #iterate through windows
#         start = round(start, 3)
#         end = round(end, 3)
#         # create column that defines whether trial lies in window or not (or is in normal condition)
#         df['in_bin'] = 0
#         df.loc[(df['actual_soa']>= start) & (df['actual_soa']<=end),'in_bin'] = 1
#         df.loc[(df['condition']=='normal') | df['condition']==0,'in_bin'] = 2
#         # calculate sdt indices for the respective window and for normal condition
#         df_sdt = df.groupby(sdt_groupby).apply(sdt) #for leaving out first 100 trials: .loc[(df['trial_nr']>102)]
#         df_sdt.reset_index(inplace=True)
#         df_sdt = df_sdt.rename(columns = {'index':'in_bin'})
#         if sdt_groupby == ['subject_id', 'block_id', 'in_bin']:
#             df_sdt = df_sdt.groupby(['subject_id', 'in_bin']).mean()
#             df_sdt.reset_index(inplace=True)
#         # get pupil difference in bin vs normal condition
#         means_p = epoch_diff_stim.loc[(epoch_diff_stim.index.get_level_values(4)>=start) & (epoch_diff_stim.index.get_level_values(4)<=end),(y>0)&(y<maxi)].groupby('subject_id').mean().max(axis=1)

#         # get correlation between sdt measure in window and sdt measure in normal condition and correct for regression towards the mean (by means of bootstrapping)
#         # true_corr, corrand, p_value = permutationTest_correlation(df_sdt.loc[(df_sdt['in_bin']==2), measure].values, df_sdt.loc[(df_sdt['in_bin']==1), measure].values, tail=tail)
#         if delta==True:
#             true_corr, p_value = sp.stats.pearsonr(means_p, df_sdt.loc[df_sdt['in_bin']==1, measure].values-df_sdt.loc[df_sdt['in_bin']==2, measure].values)
#         else:
#             true_corr, p_value = sp.stats.pearsonr(means_p, df_sdt.loc[df_sdt['in_bin']==1, measure])
#         true_corrs.append(true_corr)
#         p_values.append(p_value)
#         starts.append(start)
#         ends.append(end)
#         start += stepsize
#         end += stepsize

#     # get centers of windows and save stuff in dataframes
#     centers = [round(i + (windowsize/2),3) for i in starts]
#     # fdr correction
#     p_true , adj_p = multitest.fdrcorrection(p_values)
#     p_true = list(p_true.astype(int))

#     # plot results
#     fig, ax= plt.subplots(1,1, figsize=(3,2))
#     ax.axvline(0, color='grey', ls='dashed')
#     plt.plot(centers, true_corrs)
#     ax.set_xlabel('center of AS SOA window (windowsize={}s) in seconds relative to decision phase onset'.format(windowsize))
#     if delta == True:
#         ax.set_ylabel('cor of Δ pupil AS response with Δ {}'.format(measure))
#     else:
#         ax.set_ylabel('cor of Δ pupil AS response with {}'.format(measure))
#     # shade area that is significant after fdr correction blue
#     for i in np.where(np.diff(p_true)==1):
#         start_span = list(centers[j+1] for j in i)
#     for i in np.where(np.diff(p_true)==-1):
#         end_span = list(centers[j] for j in i)
#     if (len(start_span)!=0) or (len(end_span)!=0):
#         if (len(start_span)==0):
#             start_span.insert(0,centers[0])
#         elif (end_span[0] < start_span[0]):
#             start_span.insert(0,centers[0])
#         if (len(end_span)==0):
#             end_span.insert(0,centers[-1])
#         elif(end_span[-1] < start_span[-1]):
#             end_span.insert(len(end_span),centers[-1])
#         for i,j in zip(start_span, end_span):
#             if i == j:
#                 ax.axvline(i, alpha=0.3)
#             else:
#                 ax.axvspan((i),(j), alpha=0.3)

#     sns.despine(trim=True)
#     plt.tight_layout()

#     return fig

# #-----------------------------------------------
# ####  OLDER STUFF  ####
# #-----------------------------------------------

# # Plot for pupil resp locked to AS, choice and decision phase
# def plot_pupil_responses(epochs_p_stim, epochs_p_resp, epochs_p_dphase, groupby=['subject_id', 'condition'], ylim=None):

#     fig = plt.figure(figsize=(3.5,5.25))
#     ax = fig.add_subplot(321)
#     means = epochs_p_stim.groupby(groupby).mean().groupby(['condition']).mean()
#     sems = epochs_p_stim.groupby(groupby).mean().groupby(['condition']).sem()
#     x = np.array(means.columns, dtype=float)
#     plt.fill_between(x, means.iloc[0]-sems.iloc[0], means.iloc[0]+sems.iloc[0], color=sns.color_palette()[1], alpha=0.2)
#     plt.plot(x, means.iloc[0], color=sns.color_palette()[1],    ls='-', label='boost')
    
#     plt.axvline(0, color='k', ls='--')
#     plt.legend(loc=4)
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel('Time from stimulus (s)')
#     plt.ylabel('Pupil response (% change)')
        
#     ax = fig.add_subplot(323)
#     means = epochs_p_resp.groupby(groupby).mean().groupby(['condition']).mean()
#     sems = epochs_p_resp.groupby(groupby).mean().groupby(['condition']).sem()
#     x = np.array(means.columns, dtype=float)
#     plt.fill_between(x, means.iloc[0]-sems.iloc[0], means.iloc[0]+sems.iloc[0], color=sns.color_palette()[1], alpha=0.2)
#     plt.plot(x, means.iloc[0], color=sns.color_palette()[1],    ls='-', label='boost')
#     plt.fill_between(x, means.iloc[1]-sems.iloc[1], means.iloc[1]+sems.iloc[1], color=sns.color_palette()[0], alpha=0.2)
#     plt.plot(x, means.iloc[1], color=sns.color_palette()[0], ls='-', label='normal')
#     plt.axvspan(-1, 1.5, color='grey', alpha=0.1)
#     plt.axvline(0, color='k', ls='--')
#     plt.legend(loc=4)
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel('Time from choice (s)')
#     plt.ylabel('Pupil response (% change)')

#     ax = fig.add_subplot(324)
#     means = epochs_p_resp.groupby(groupby).mean().groupby(['condition']).mean()
#     sems = epochs_p_resp.groupby(groupby).mean().groupby(['condition']).sem()
#     x = np.array(means.columns, dtype=float)
#     mean = (means.iloc[0]-means.iloc[1])/2
#     sem = (sems.iloc[0]+sems.iloc[1])/2
#     plt.fill_between(x, mean-sem, mean+sem, color='grey', alpha=0.2)
#     plt.plot(x, mean, color='grey',    ls='-', label='difference')
#     plt.axvline(0, color='k', ls='--')
#     plt.legend(loc=4)
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel('Time from choice (s)')
#     plt.ylabel('Pupil response (% change)')

#     ax = fig.add_subplot(325)
#     means = epochs_p_dphase.groupby(groupby).mean().groupby(['condition']).mean()
#     sems = epochs_p_dphase.groupby(groupby).mean().groupby(['condition']).sem()
#     x = np.array(means.columns, dtype=float)
#     plt.fill_between(x, means.iloc[0]-sems.iloc[0], means.iloc[0]+sems.iloc[0], color=sns.color_palette()[1], alpha=0.2)
#     plt.plot(x, means.iloc[0], color=sns.color_palette()[1],    ls='-', label='boost')
#     plt.fill_between(x, means.iloc[1]-sems.iloc[1], means.iloc[1]+sems.iloc[1], color=sns.color_palette()[0], alpha=0.2)
#     plt.plot(x, means.iloc[1], color=sns.color_palette()[0], ls='-', label='normal')
#     #soas = epochs_p_dphase.index.get_level_values(4).values #!work
#     #y = [-10]*len(soas)
#     #plt.scatter(epochs_p_dphase.index.get_level_values(4).values, y, s=2)
#     #plt.plot(soas, kind='density')
#     plt.axvspan(-1, 1.5, color='grey', alpha=0.1)
#     plt.axvline(0, color='k', ls='--')
#     plt.legend(loc=4)
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel('Time from decision phase onset (s)')
#     plt.ylabel('Pupil response (% change)')

#     ax = fig.add_subplot(326)
#     means = epochs_p_dphase.groupby(groupby).mean().groupby(['condition']).mean()
#     sems = epochs_p_dphase.groupby(groupby).mean().groupby(['condition']).sem()
#     x = np.array(means.columns, dtype=float)
#     mean = (means.iloc[0]-means.iloc[1])/2
#     sem = (sems.iloc[0]+sems.iloc[1])/2
#     plt.fill_between(x, mean-sem, mean+sem, color='grey', alpha=0.2)
#     plt.plot(x, mean, color='grey',    ls='-', label='difference')
#     plt.axvline(0, color='k', ls='--')
#     plt.legend(loc=4)
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel('Time from decision phase onset (s)')
#     plt.ylabel('Pupil response (% change)')
    
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# # non heat map plot of pupil response by bin
# def plot_pupil_by_bin(epochs_p_stim_b, groupby=['subject_id'], bin_nrs=10):
#     if bin_nrs==10:
#         binning=['(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'] 
#     fig = plt.figure(figsize=(9,10.5))
#     for soa_bin, new_df in epochs_p_stim_b.groupby('soa_bin'):
#         i=int(soa_bin) +1
#         ax = fig.add_subplot(4,3,i)
#         means = new_df.groupby(groupby).mean().mean()
#         sems = new_df.groupby(groupby).mean().sem()
#         x = np.array(means.index, dtype=float)
#         plt.fill_between(x, means-sems, means+sems, color=sns.color_palette()[1], alpha=0.2)
#         plt.plot(x, means, color=sns.color_palette()[1], ls='-', lw=0.8)
#         # ax.set_yticks(np.arange()) #TODO
        
#         plt.axvline(0, color='k', ls='--')
#         # plt.legend(loc=4)
#         plt.ylim((-3,5))
#         plt.xlabel('Time(s) from AS in bin {}'.format(binning[(int(soa_bin-1))]))
#         plt.ylabel('Pupil response (% change)')
            
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# # non heat map plot of pupil response diff between normal cond and bins
# def plot_pupil_diff_bin(epochs_diff_dphase, groupby=['subject_id'], bin_nrs=10):
#     if bin_nrs==10:
#         binning=['(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'] 
#     fig = plt.figure(figsize=(6,5))
#     plt.axvline(0, color='k', ls='--', linewidth=0.5)
#     patch = []
#     for soa_bin, new_df in epochs_diff_dphase.groupby('soa_bin'):
#         i=int(soa_bin) 
#         if i > 0:   
#             means = new_df.groupby(groupby).mean().mean()
#             sems = new_df.groupby(groupby).mean().sem()
#             x = np.array(means.index, dtype=float)
#             plt.fill_between(x, means-sems, means+sems, color=sns.color_palette("crest", 11)[i], alpha=0.2)
#             plt.plot(x, means, color=sns.color_palette("crest", 11)[i], ls='-', lw=0.8)
#             # ax.set_yticks(np.arange()) #TODO
#             patch.append(mpatches.Patch(color=sns.color_palette("crest", 11)[i], label= binning[i-1]))

#     fig.legend(title= 'AS SOA bin (s)', title_fontsize = 9, handles=patch, loc='upper left', bbox_to_anchor=(1, 1))    
#     plt.ylim((-2,4))
#     plt.xlabel('Time(s) from decision phase onset', fontsize=9)
#     plt.ylabel('Difference in pupil response (% change)', fontsize=9)
             
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# def plot_pupil_diff_sliding_bin(epochs_diff_dphase, windowsize, stepsize, groupby=['subject_id'], bin_nrs=10, ylim=(-2,4)):
#     # if bin_nrs==10:
#     #     binning=['(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'] 
#     fig = plt.figure(figsize=(6,5))
#     plt.axvline(0, color='k', ls='--', linewidth=0.5)
#     patch = []
#     iter = np.arange(0,int(((3500-(windowsize * 1000))/(stepsize * 1000))+1),1) #526 --> was that correct? shouldn't it have been 525?
#     start = float(-3)
#     end = float(-3+windowsize)
#     starts = []
#     ends = []
#     for i in iter:
#         start = round(start, 3)
#         end = round(end, 3)
#         means = epochs_diff_dphase.loc[(epochs_diff_dphase.index.get_level_values(4)>=start) & (epochs_diff_dphase.index.get_level_values(4)<=end)].groupby(groupby).mean().mean()
#         sems = epochs_diff_dphase.loc[(epochs_diff_dphase.index.get_level_values(4)>=start) & (epochs_diff_dphase.index.get_level_values(4)<=end)].groupby(groupby).mean().sem()
#         x = np.array(means.index, dtype=float)
#         plt.fill_between(x, means-sems, means+sems, color=sns.color_palette("crest", 11)[i+1], alpha=0.2)
#         plt.plot(x, means, color=sns.color_palette("crest", 11)[i+1], ls='-', lw=0.8)
#         # ax.set_yticks(np.arange()) #TODO
#         patch.append(mpatches.Patch(color=sns.color_palette("crest", 11)[i+1], label= '({})-({})'.format(start,end)))
#         start += stepsize
#         end += stepsize

#     fig.legend(title= 'AS SOA bin (s)', title_fontsize = 9, handles=patch, loc='upper left', bbox_to_anchor=(1, 1))    
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel('Time(s) from decision phase onset', fontsize=9)
#     plt.ylabel('Difference in pupil response (% change)', fontsize=9)
             
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# def plot_tpr_stats_sliding_bin(df, windowsize, stepsize, groupby=['subject_id'], ylim=(-2,4)):
    
#     means_normal = df.loc[(df['condition']==0)].groupby(['subject_id']).tpr.mean()
    
#     fig = plt.figure(figsize=(3,2))
#     ax = fig.add_subplot(111)
#     plt.axhline(0, color='k', ls='--', linewidth=0.5)
#     sns.stripplot(x=0, y=means_normal.values-means_normal.values, color=sns.color_palette("crest", 11)[0+1])
#     patch = []
#     iter = np.arange(0,int(((3500-(windowsize * 1000))/(stepsize * 1000))+1),1) #526 --> was that correct? shouldn't it have been 525?
#     start = float(-3)
#     end = float(-3+windowsize)
#     for i in iter:
#         start = round(start, 3)
#         end = round(end, 3)
#         means = df.loc[(df['actual_soa']>=start) & (df['actual_soa']<=end)].groupby(groupby).tpr.mean()
#         sns.stripplot(x=i+1, y=means.values-means_normal.values, color=sns.color_palette("crest", 11)[i+1])
#         # ax.set_yticks(np.arange()) #TODO
#         patch.append(mpatches.Patch(color=sns.color_palette("crest", 11)[i+1], label= '({})-({})'.format(start,end)))
#         start += stepsize
#         end += stepsize

#     fig.legend(title= 'AS SOA bin (s)', title_fontsize = 7, handles=patch, loc='upper left', bbox_to_anchor=(1, 1))    
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel('Time(s) from decision phase onset')
#     plt.ylabel('Δ Pupil response (% change)')
             
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# # non heat map plot of pupil response by bin for each subject
# def plot_pupil_bin_singles(epochs_p_stim_b, groupby=None, bin_nrs=10):
#     if bin_nrs==10:
#         binning=['(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'] 
#     fig = plt.figure(figsize=(18,21))
#     for sub_id, df in epochs_p_stim_b.groupby('subject_id'):
#         ax = fig.add_subplot(7,6,sub_id)
#         plt.axvline(0, color='k', ls='--')

#         for soa_bin, new_df in df.groupby('soa_bin'):
#             if groupby is not None:
#                 means = new_df.groupby(groupby).mean().mean()
#             else:
#                 means = new_df.mean()
#             #sems = new_df.groupby(groupby).mean().sem()
#             x = np.array(means.index, dtype=float)
#             #plt.fill_between(x, means-sems, means+sems, color=sns.color_palette()[int(soa_bin)], alpha=0.2)
#             plt.plot(x, means, color=sns.color_palette("vlag",12)[int(soa_bin)], ls='-', lw=1.5, label=binning[int(soa_bin)-1])
#             # ax.set_yticks(np.arange()) #TODO
            
#         plt.ylim((-10,12))
#         plt.xlabel('Time(s) from AS in bin {}'.format(binning[(int(soa_bin-1))]))
#         plt.ylabel('Pupil response (% change)')

        
#     plt.legend(bbox_to_anchor=(1,1), loc="upper left")
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# # Plot for pupil resp for each subject (choose locking by epoch input, make ax label by which_plot)
# def plot_pupil_responses_all(epoch_p, which_plot, groupby=['block_id','trial_nr'], ylim=None):

#     fig = plt.figure(figsize=(9,10.5))
#     dict = {} #commented stuff in this function is for additionally plotting the peaks found with the find_peaks function from scipy
#     for sub_id, new_df in epoch_p.groupby('subject_id'):
#         #i=int("76"+str(sub_id))
#         ax = fig.add_subplot(7,6,sub_id)
#         means = new_df.groupby(groupby).mean().mean()
#         sems = new_df.groupby(groupby).mean().sem()
#         x = np.array(means.index, dtype=float)
#         plt.fill_between(x, means-sems, means+sems, color=sns.color_palette()[1], alpha=0.2)
#         plt.plot(x, means, color=sns.color_palette()[1], ls='-', lw=0.8)
#         # ax.set_yticks(np.arange()) #TODO
#         # peaks, _ = find_peaks(means)
#         # dict[str(sub_id)]=pd.DataFrame(list(zip(x[peaks],means[peaks])), columns=['peak_ind', 'peak_val'])
#         # plt.plot(x[peaks], means[peaks], "x")
#         # ax.set_xticks([0,1,2,3,4,5])

#         plt.axvline(0, color='k', ls='--')
#         # plt.legend(loc=4)
#         if ylim is not None:
#             plt.ylim(ylim)
#         if which_plot=='stims':
#             plt.xlabel('Time from AS (s)')
#         if which_plot=='resps':
#             plt.xlabel('Time from response onset(s)')
#         if which_plot=='dphases':
#             plt.xlabel('Time from decision onset (s)')
#         plt.ylabel('Pupil response (% change)')
            
       
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig # , dict

# # Plot for comparing AS pupil response
# def plot_compare_p_resp(epoch_p, epoch_t, which_plot, groupby=['block_id','trial_nr'], ylim=None):

#     fig = plt.figure(figsize=(9,10.5))
#     for ((sub_id, new_df), (sub_id2, new_df2)) in zip(epoch_p.groupby('subject_id'), epoch_t.groupby('subject_id')):
#         #i=int("76"+str(sub_id))
#         ax = fig.add_subplot(7,6,sub_id)
#         means = new_df.groupby(groupby).mean().mean()
#         sems = new_df.groupby(groupby).mean().sem()
#         x = np.array(means.index, dtype=float)
#         plt.fill_between(x, means-sems, means+sems, color=sns.color_palette()[1], alpha=0.2)
#         plt.plot(x, means, color=sns.color_palette()[1], ls='-', lw=0.8) #, label='passive task')
#         # ax.set_yticks(np.arange()) #TODO
        
#         means2 = new_df2.groupby(groupby).mean().mean()
#         sems2 = new_df2.groupby(groupby).mean().sem()
#         x2 = np.array(means2.index, dtype=float)
#         plt.fill_between(x2, means2-sems2, means2+sems2, color=sns.color_palette()[0], alpha=0.2)
#         plt.plot(x2, means2, color=sns.color_palette()[0], ls='-', lw=0.8) #, label='main task')

#         # plt.plot(x2, means2-means, color=sns.color_palette()[2], ls='-', lw=0.8) #for plotting the difference between the two first lines

#         plt.axvline(0, color='k', ls='--')
#         # plt.legend(loc=1)
#         if ylim is not None:
#             plt.ylim(ylim)
#         if which_plot=='stims':
#             plt.xlabel('Time from AS (s)')
#         if which_plot=='resps':
#             plt.xlabel('Time from response onset(s)')
#         if which_plot=='dphases':
#             plt.xlabel('Time from decision onset (s)')
#         plt.ylabel('Pupil response (% change)')
            
#     orange_patch = mpatches.Patch(color=sns.color_palette()[1], label='passive task')
#     blue_patch = mpatches.Patch(color=sns.color_palette()[0], label='main task')
#     fig.legend(loc=1, handles=[orange_patch, blue_patch])
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# # Boxplots for reaction times
# def plot_rt(df_meta, groupby):
    
#     fig, ax= plt.subplots(1,1)
#     sns.boxplot(x=df_meta[groupby], y=df_meta['rt'], ax=ax)
#     ax.set_ylabel('reaction time (in s)', fontsize=9)
#     if (groupby == 'soa_bin') & (os.getcwd() == 'C:\\Users\\Josefine\\Documents\\Promotion\\arousal_memory_experiments\\data\\pilot2_contrast_detection'):
#         ax.set_xlabel('SOA bin of auditory stimulus (in s; locked to decision phase onset)', fontsize=9)
#         ax.set_xticklabels(['no stim','(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15'], rotation=50)
#     elif groupby=='soa_bin':
#         ax.set_xlabel('SOA bin of auditory stimulus (in s; locked to decision phase onset)', fontsize=9)
#         ax.set_xticklabels(['no stim','(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'], rotation=50)

#     return fig

# def plot_scatter_rt(df_meta_whole, groupby=['subject_id', 'block_id'], bin_nrs=10, ylim=None):

#     if bin_nrs==10:
#         binning=['(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'] 

#     fig = plt.figure(figsize=(9,10.5))
#     for_diff = np.array(df_meta_whole.loc[df_meta_whole['condition']=='normal'].groupby('subject_id').rt.mean(), dtype=float)
#     # df_meta_whole['correctness']=np.NaN
#     # df_meta_whole.loc[(df_meta_whole['sdt']==1) | (df_meta_whole['sdt']==4), 'correctness'] = 1
#     # df_meta_whole.loc[(df_meta_whole['sdt']==2) | (df_meta_whole['sdt']==3), 'correctness'] = 0
#     # x_for_diff = np.array(df_meta_whole.loc[df_meta_whole['condition']=='normal'].groupby('subject_id').correctness.mean(), dtype=float)

#     for soa_bin, new_df in df_meta_whole.groupby('soa_bin'):
#         if soa_bin>0:
#             i=int(soa_bin)
#             ax = fig.add_subplot(4,3,i)
#             y = np.array(new_df.groupby('subject_id').rt.mean(), dtype=float)
#             y_diff=y-for_diff
#             # x = np.array(new_df.groupby('subject_id').correctness.mean(), dtype=float)
#             # x_diff=x-x_for_diff
#             plt.scatter(for_diff, y_diff, color=sns.color_palette("cubehelix",41), ls='-', lw=0.8) # or x_diff
#             plt.xlabel('Mean RT normal condition' ) #or: Delta accuracy
#             plt.ylabel('Delta mean RTs; bin {}'.format(binning[(int(soa_bin-1))]) )
#             if ylim is not None:
#                 plt.ylim(ylim)
#             plt.axhline(0, color='k', ls='--')
#             plt.axvline(0, color='k', ls='--')

            
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# # Violin plots by bin choose binning, and grouping and variable (e.g. rt, d or c)
# def violinplot(df, groupby, x, y, z, ylim=None):
    
#     grouping=[groupby, x, z]
#     means = df.groupby(grouping).mean()
#     means= means.reset_index()

#     fig, ax= plt.subplots(1,1)
#     if ylim is not None:
#         plt.ylim(ylim)
#     sns.violinplot(data=means, x=x, y=y ,hue=z, palette="pastel", split=True, linewidth=0.7, alpha=.1, ax=ax)
#     sns.stripplot(data=means, x=x, y=y, hue=z, dodge=True, s=4, linewidth=0.7, alpha=0.6, legend=False, ax=ax)
#     ax.set_ylabel('reaction time (in s)', fontsize=9)
#     if x=='soa_bin':
#         ax.set_xlabel('SOA bin of auditory stimulus (in s; locked to decision phase onset)', fontsize=9)
#         ax.set_xticklabels(['no stim','(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'], rotation=50)
#     if x=='resp_soa_bin':
#         ax.set_xlabel('SOA bin of auditory stimulus (in s; locked to response)', fontsize=9)
#         ax.set_xticklabels(['no stim','(-10)-(-5)', '(-5)-(-4.5)', '(-4.5)-(-4)', '(-4)-(-3.5)', '(-3.5)-(-3)', '(-3)-(-2.5)', '(-2.5)-(-2)', '(-2)-(-1.5)', '(-1.5)-(-1)', '(-1)-(-0.5)', '(-0.5)-0'], rotation=50)
#     if z=='correctness':
#         handles, labels = ax.get_legend_handles_labels()
#         labels = ["incorrect", "correct"]
#         ax.legend(handles, labels)

#     sns.despine(trim=True)
#     plt.tight_layout()

#     return fig

# # Swarmplots by bin choose binning, and grouping and variable (e.g. rt, d or c)
# def swarmplot(df, groupby, x, y, z, ylim=None):
    
#     grouping=[groupby, x, z]
#     means = df.groupby(grouping).mean()
#     means= means.reset_index()

#     fig, ax= plt.subplots(1,1)
#     if ylim is not None:
#         plt.ylim(ylim)
#     sns.violinplot(data=means, x=x, y=y, color='w', linewidth=0.7, ax=ax)
#     sns.swarmplot(data=means, x=x, y=y, hue=z, palette='Blues', dodge=True, s=4, linewidth=0.7, ax=ax)
#     ax.set_ylabel('reaction time (in s)', fontsize=9)
#     if x=='soa_bin':
#         ax.set_xlabel('SOA bin of auditory stimulus (in s; locked to decision phase onset)', fontsize=9)
#         ax.set_xticklabels(['no stim','(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'], rotation=50)
#     if x=='resp_soa_bin':
#         ax.set_xlabel('SOA bin of auditory stimulus (in s; locked to response)', fontsize=9)
#         ax.set_xticklabels(['no stim','(-10)-(-5)', '(-5)-(-4.5)', '(-4.5)-(-4)', '(-4)-(-3.5)', '(-3.5)-(-3)', '(-3)-(-2.5)', '(-2.5)-(-2)', '(-2)-(-1.5)', '(-1.5)-(-1)', '(-1)-(-0.5)', '(-0.5)-0'], rotation=50)
#     if z=='correctness':
#         handles, labels = ax.get_legend_handles_labels()
#         labels = ["incorrect", "correct"]
#         ax.legend(handles, labels)
#     if z=='peak_max_sub':
#         titl = 'indiv max pupil peak'
#     if z=='peak_first_sub':
#         titl = 'indiv first pupil peak'
#     if z=='mean_p_sub1':
#         titl = 'indiv mean pupil resp'
#     if z=='max_p_sub1':
#         titl = 'indiv max pupil resp'
#     if z=='max_p_sub':
#         titl = 'indiv max pupil resp (diff)'
#     if z=='mean_p_sub':
#         titl = 'indiv mean pupil resp (diff)'

#     plt.legend(title=titl, title_fontsize=9,loc='upper left', bbox_to_anchor=(1, 1))
#     sns.despine(trim=True)
#     plt.tight_layout()

#     return fig

# def swarmplot_sdt(df_sdt_bin, groupby, var, z, ylim=None):
    
#     fig, ax= plt.subplots(1,1)
#     sns.violinplot(data=df_sdt_bin, x=groupby, y=var, color='w', linewidth=0.7, ax=ax)
#     sns.swarmplot(data=df_sdt_bin, x=groupby, y=var, hue=z, palette='Blues', dodge=True, s=4, linewidth=0.7, ax=ax)

#     ax.set_ylabel(var, fontsize=9)
#     if ylim is not None:
#         plt.ylim(ylim)
#     if groupby=='soa_bin':
#         ax.set_xlabel('SOA bin of auditory stimulus (in s; locked to decision phase onset)', fontsize=9)
#         ax.set_xticklabels(['no stim','(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'], rotation=50)
#     if groupby=='resp_soa_bin':
#         ax.set_xlabel('SOA bin of auditory stimulus (in s; locked to response)', fontsize=9)
#         ax.set_xticklabels(['no stim','(-10)-(-5)', '(-5)-(-4.5)', '(-4.5)-(-4)', '(-4)-(-3.5)', '(-3.5)-(-3)', '(-3)-(-2.5)', '(-2.5)-(-2)', '(-2)-(-1.5)', '(-1.5)-(-1)', '(-1)-(-0.5)', '(-0.5)-0'], rotation=50)
#     if groupby=='big_bin':
#         ax.set_xlabel('SOA bin of auditory stimulus (in s; locked to decision phase onset)', fontsize=9)
#         ax.set_xticklabels(['no stim','(-3)-(-2.125)', '(-2.125)-(-1.25)', '(-1.25)-(-0.375)', '(-0.375)-0.5'], rotation=50)

#     if z=='peak_max_sub':
#         titl = 'indiv max pupil peak'
#     if z=='peak_first_sub':
#         titl = 'indiv first pupil peak'
#     if z=='mean_p_sub1':
#         titl = 'indiv mean pupil resp'
#     if z=='max_p_sub1':
#         titl = 'indiv max pupil resp'
#     if z=='max_p_sub':
#         titl = 'indiv max pupil resp (diff)'
#     if z=='mean_p_sub':
#         titl = 'indiv mean pupil resp (diff)'

#     if z is not None:
#         plt.legend(title=titl, title_fontsize=9,loc='upper left', bbox_to_anchor=(1, 1))
#     sns.despine(trim=True)
#     plt.tight_layout()

#     return fig

# def plot_tpr_sdt(df_sdt_bin, groupby, var, ylim=None, xlim=None):
    
#     fig, ax= plt.subplots(1,1)
#     x = range((int(max(df_sdt_bin[groupby]))+1))
#     sns.violinplot(data=df_sdt_bin, x=groupby, y=var, color='w', linewidth=0.7, ax=ax)
#     sns.swarmplot(data=df_sdt_bin, x=groupby, y=var, s=4, linewidth=0.7, ax=ax)
#     for s in df_sdt_bin['subject_id']:
#         plt.plot(x, df_sdt_bin.loc[(df_sdt_bin['subject_id']==s), var], color='0.8', ls='--', lw=0.3)

#     md = smf.mixedlm("{} ~ {}".format(var, groupby), df_sdt_bin, groups=df_sdt_bin["subject_id"]) #FIXME wrong usage of "groups"?
#     mdf = md.fit(method=["lbfgs"])
#     print(mdf.summary())    
#     y = mdf.params[0] + mdf.params[1]*x

#     plt.plot(x,y)

#     ax.set_ylabel(var, fontsize=9)
#     if ylim is not None:
#         plt.ylim(ylim)
#     if xlim is not None:
#         plt.xlim(xlim)
#     if groupby=='pupil_tpr_4bin':
#         ax.set_xlabel('TPR bin', fontsize=9)
#         ax.set_xticklabels([1,2,3,4])

#     sns.despine(trim=True)
#     plt.tight_layout()

#     return fig


# def scatter_sdt_pupil(df_sdt, x, y, y_type):
#     fig, ax= plt.subplots(1,1)
#     y1 = df_sdt.loc[(df_sdt['condition']=='boost')].reset_index()
#     y2 = df_sdt.loc[(df_sdt['condition']=='normal')].reset_index()
#     y1['diff'] = y1[y] - y2[y]
#     y1['diff_abs'] = abs(y1['diff'].copy())
#     if y_type == 'absolute':
#         # plt.scatter(y1[x],y1['diff_abs'], color='grey')
#         sns.regplot(data=y1, x=x, y='diff_abs', ax=ax, ci = None)
#         # c, i = np.polyfit(x=y1[x],y=y1['diff_abs'], deg=1)
#         # ax.plot(y1[x], i + c * y1[x], '--')
#         r = stats.pearsonr(y1[x],y1['diff_abs'])
#     if y_type == 'plain':
#         # plt.scatter(y1[x],y1['diff'], color='grey')
#         sns.regplot(data=y1, x=x, y='diff', ax=ax, ci= None)
#         r = stats.pearsonr(y1[x],y1['diff'])

#     ax.text(6, 0.05, 'r = {}, p = {}'.format(round(r[0],2),round(r[1],2)), fontsize=9)

#     if x=='peak_max_sub':
#         ax.set_xlabel('individual global peak in pupil response within 2.5s after AS', fontsize=9)
#     if x=='peak_first_sub':
#         ax.set_xlabel('individual first peak in pupil response after AS', fontsize=9)
#     if x=='mean_p_sub1':
#         ax.set_xlabel('individual mean pupil response within 5s after AS', fontsize=9)
#     if x=='max_p_sub1':
#         ax.set_xlabel('individual max pupil response within 5s after AS', fontsize=9)
#     if x=='max_p_sub':
#         ax.set_xlabel('individual max pupil response (compared to normal trials) within 5s after AS', fontsize=9)
#     if x=='mean_p_sub':
#         ax.set_xlabel('individual mean pupil response (compared to normal trials) within 5s after AS', fontsize=9)
#     if (y == 'c') and (y_type == 'plain'):
#         ax.set_ylabel('difference in criterion boost-normal trials', fontsize=9)
#     if (y == 'c') and (y_type == 'absolute'):
#         ax.set_ylabel('strength of difference in criterion boost-normal trials', fontsize=9)
#     if y == 'c_abs':
#         ax.set_ylabel('difference in absolute criterion boost-normal trials', fontsize=9)
#     if y == 'd':
#         ax.set_ylabel("difference in d' boost-normal trials", fontsize=9)
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# def scatter_delta_c_pupil(df_sdt, x, y, z, groupby='condition', level1='normal', level2='boost', y_type='plain'):
#     fig, ax= plt.subplots(1,1)
#     y1 = df_sdt.loc[(df_sdt[groupby]==level2)].reset_index()
#     y2 = df_sdt.loc[(df_sdt[groupby]==level1)].reset_index()
#     y2['diff'] = y1[y] - y2[y]
#     y2['diff_abs'] = abs(y2['diff'].copy())
    
#     plt.axhline(0, color='gray', lw=0.8, ls='dashed')

#     if y_type == 'plain':
#         sns.scatterplot(data=y2, x=x, y='diff', hue=z)
#         sns.regplot(data=y2, x=x, y='diff', ax=ax, ci= None, scatter=False)
#         truecorr, corrrand, p_value = permutationTest_correlation(df_sdt.loc[(df_sdt[groupby]==level1), y].values, df_sdt.loc[(df_sdt[groupby]==level2), y].values, tail=-1)

#     ax.text(1.5, 0.05, 'r = {}, p = {}'.format(round(truecorr,2),round(p_value,2)), fontsize=9)

#     if z=='peak_max_sub':
#         titl = 'max pupil peak'
#     if z=='peak_first_sub':
#         titl = 'first pupil peak'
#     if z=='mean_p_sub1':
#         titl = 'mean pupil resp'
#     if z=='max_p_sub1':
#         titl = 'indiv max pupil resp'
#     if z=='max_p_sub':
#         titl = 'indiv max pupil resp (diff)'
#     if z=='mean_p_sub':
#         titl = 'indiv mean pupil resp (diff)'

#     if (y == 'c') and (y_type == 'plain'):
#         if level2=='boost':
#             ax.set_ylabel('difference in criterion boost-normal trials', fontsize=9)
#         if level2==1:
#             ax.set_ylabel('difference in criterion boost(bin 1: (-3) - (-2.125)s)-normal trials', fontsize=9)
#         if level2==2:
#             ax.set_ylabel('difference in criterion boost(bin 2: (-2.125) - (-1.25)s)-normal trials', fontsize=9)
#         if level2==3:
#             ax.set_ylabel('difference in criterion boost(bin 3: (-1.25) - (-0.375)s)-normal trials', fontsize=9)
#         if level2==4:
#             ax.set_ylabel('difference in criterion boost(bin 4: (-1.25) - 0.5s)-normal trials', fontsize=9)
#     if (y == 'c') and (y_type == 'absolute'):
#         ax.set_ylabel('strength of difference in criterion boost-normal trials', fontsize=9)
#     if y == 'c_abs':
#         ax.set_ylabel('difference in absolute criterion boost-normal trials', fontsize=9)
#     if y == 'd':
#         ax.set_ylabel("difference in d' boost-normal trials", fontsize=9)
#     if x == 'c':
#         ax.set_xlabel('criterion on normal trials', fontsize=9)
#     if x == 'c_abs':
#         ax.set_xlabel('absolute criterion on normal trials', fontsize=9)
#     if x == 'd':
#         ax.set_xlabel("d' on normal trials", fontsize=9)
    
#     plt.legend(title=titl, title_fontsize=9,loc='upper right')
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# # plot within subject differences in delta c between low AS pupil response trials (compared to normal) and high AS pupil response trials (compared to normal)
# def scatter_delta_c_p_split(df_sdt_cond, df_sdt_p_split, x, y, z, groupby1='condition', groupby2='p_mean_split', level1='normal', level2=0, level3=1, y_type='plain'):
#     fig, ax= plt.subplots(1,1)
#     y1 = df_sdt_cond.loc[(df_sdt_cond[groupby1]==level1)].reset_index()
#     y2 = df_sdt_p_split.loc[(df_sdt_p_split[groupby2]==level2)].reset_index()
#     y3 = df_sdt_p_split.loc[(df_sdt_p_split[groupby2]==level3)].reset_index()
#     y2['diff'] = y2[y] - y1[y]
#     y3['diff'] = y3[y] - y1[y]
#     y2['diff_abs'] = abs(y2['diff'].copy())
#     y3['diff_abs'] = abs(y2['diff'].copy())
#     y2 = pd.concat([y2, y3])
    
#     plt.axhline(0, color='gray', lw=0.8, ls='dashed')

#     if y_type == 'plain':
#         sns.scatterplot(data=y2, x=x, y='diff') #, hue=z)
#         sns.regplot(data=y2, x=x, y='diff', ax=ax, ci= None, scatter=False)
#         truecorr, corrrand, p_value = permutationTest_correlation(df_sdt_cond.loc[(df_sdt_cond[groupby]==level1), y].values, df_sdt.loc[(df_sdt[groupby]==level2), y].values, tail=-1) #TODO fix

#     ax.text(1.5, 0.05, 'r = {}, p = {}'.format(round(truecorr,2),round(p_value,2)), fontsize=9)

#     if z=='peak_max_sub':
#         titl = 'max pupil peak'
#     if z=='peak_first_sub':
#         titl = 'first pupil peak'
#     if z=='mean_p_sub1':
#         titl = 'mean pupil resp'
#     if z=='max_p_sub1':
#         titl = 'indiv max pupil resp'
#     if z=='max_p_sub':
#         titl = 'indiv max pupil resp (diff)'
#     if z=='mean_p_sub':
#         titl = 'indiv mean pupil resp (diff)'

#     if (y == 'c') and (y_type == 'plain'):
#         if level2=='boost':
#             ax.set_ylabel('difference in criterion boost-normal trials', fontsize=9)
#         if level2==1:
#             ax.set_ylabel('difference in criterion boost(bin 1: (-3) - (-2.125)s)-normal trials', fontsize=9)
#         if level2==2:
#             ax.set_ylabel('difference in criterion boost(bin 2: (-2.125) - (-1.25)s)-normal trials', fontsize=9)
#         if level2==3:
#             ax.set_ylabel('difference in criterion boost(bin 3: (-1.25) - (-0.375)s)-normal trials', fontsize=9)
#         if level2==4:
#             ax.set_ylabel('difference in criterion boost(bin 4: (-1.25) - 0.5s)-normal trials', fontsize=9)
#     if (y == 'c') and (y_type == 'absolute'):
#         ax.set_ylabel('strength of difference in criterion boost-normal trials', fontsize=9)
#     if y == 'c_abs':
#         ax.set_ylabel('difference in absolute criterion boost-normal trials', fontsize=9)
#     if y == 'd':
#         ax.set_ylabel("difference in d' boost-normal trials", fontsize=9)
#     if x == 'c':
#         ax.set_xlabel('criterion on normal trials', fontsize=9)
#     if x == 'c_abs':
#         ax.set_xlabel('absolute criterion on normal trials', fontsize=9)
#     if x == 'd':
#         ax.set_xlabel("d' on normal trials", fontsize=9)
    
#     plt.legend(title=titl, title_fontsize=9,loc='upper right')
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# def plot_sdt(df_sdt_bin, groupby, var):
    
#     fig, ax= plt.subplots(1,1)
#     sns.swarmplot(x=df_sdt_bin[groupby], y=df_sdt_bin[var], ax=ax, hue=df_sdt_bin['subject_id'], palette='tab10')
#     sns.violinplot(x=df_sdt_bin[groupby], y=df_sdt_bin[var], ax=ax, color='w')
#     ax.set_ylabel(var, fontsize=9)
#     if (groupby == 'soa_bin') & (os.getcwd() == 'C:\\Users\\Josefine\\Documents\\Promotion\\arousal_memory_experiments\\data\\pilot2_contrast_detection'):
#         ax.set_xlabel('SOA bin of auditory stimulus (in s; locked to decision phase onset)', fontsize=9)
#         ax.set_xticklabels(['no stim','(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15'], rotation=50)
#     elif groupby=='soa_bin':
#         ax.set_xlabel('SOA bin of auditory stimulus (in s; locked to decision phase onset)', fontsize=9)
#         ax.set_xticklabels(['no stim','(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'], rotation=50)
#     if groupby=='resp_soa_bin':
#         ax.set_xlabel('SOA bin of auditory stimulus (in s; locked to response)', fontsize=9)
#         ax.set_xticklabels(['no stim','(-10)-(-5)', '(-5)-(-4.5)', '(-4.5)-(-4)', '(-4)-(-3.5)', '(-3.5)-(-3)', '(-3)-(-2.5)', '(-2.5)-(-2)', '(-2)-(-1.5)', '(-1.5)-(-1)', '(-1)-(-0.5)', '(-0.5)-0'], rotation=50)

#     sns.despine(trim=True)
#     plt.tight_layout()

#     return fig

# def plot_rt_sliding_bin(df_meta_whole, windowsize, stepsize, groupby=['subject_id', 'block_id'], ylim=(-2,4)): # careful: groupby has to be same as when creating df_sdt_bin!
#     fig, ax= plt.subplots(1,1)
#     means1 = df_meta_whole.loc[(df_meta_whole['soa_bin']==0)].groupby(groupby).rt.mean()
#     sns.swarmplot(x= 0, y=means1, ax=ax)
#     plt.violinplot(means1, positions= [0])

#     iter = np.arange(0,int(((3500-(windowsize * 1000))/(stepsize * 1000))+1),1) #526 --> was that correct? shouldn't it have been 525?
#     start = float(-3)
#     end = float(-3+windowsize)
#     starts = []
#     ends = []
#     for i in iter:
#         start = round(start, 3)
#         end = round(end, 3)
#         df_meta_whole['in_bin'] = 0
#         df_meta_whole.loc[(df_meta_whole['actual_soa']>= start) & (df_meta_whole['actual_soa']<=end),'in_bin'] = 1
#         df_meta_whole.loc[(df_meta_whole['condition']=='normal'),'in_bin'] = 2
#         means = df_meta_whole.loc[(df_meta_whole['in_bin']==1)].groupby(groupby).rt.mean() #for leaving out first 100 trials: .loc[(df_meta_whole['trial_nr']>102)]
#         sns.swarmplot(x= (i+1), y= means, ax=ax)
#         plt.violinplot(means, positions=[(i+1)])

#         starts.append(start)
#         ends.append(end)
#         start += stepsize
#         end += stepsize

#     centers = [str(round(i + (windowsize/2),3)) for i in starts]
#     centers.insert(0,'no stim')
#     ax.set_xticklabels(centers)
#     ax.set_xlabel('center of AS SOA window (windowsize={}s) in seconds relative to decision phase onset'.format(windowsize), fontsize=8)
#     ax.set_ylabel('cor of RT with Δ RT (AS SOA window - no-AS condition)', fontsize=8)
#     sns.despine(trim=True)
#     plt.tight_layout()

#     return fig

# # Heat map for each bin's pupil resp locked to decision phase onset and choice
# def plot_pupil_bin_responses(epochs_p_resp, epochs_p_dphase, x, z, groupby=['subject_id', 'soa_bin'], ylim=None):

#     means_dphase = epochs_p_dphase.groupby(groupby).mean().groupby(['soa_bin']).mean()
#     means_resp = epochs_p_resp.groupby(groupby).mean().groupby(['soa_bin']).mean()

#     fig, ax= plt.subplots(1,2, figsize=(14,5))
#     fig.suptitle('Pupil response by SOA of auditory stimulus')
#     ax[0].set_title('locked to decision phase onset', fontsize=11)
#     ax[1].set_title('locked to response', fontsize=11)
#     sns.heatmap(means_dphase, ax=ax[0], cbar_kws={'label': 'pupil response in percent change'})
#     ax[0].figure.axes[-1].yaxis.label.set_size(9)
#     sns.heatmap(means_resp, ax=ax[1], cbar_kws={'label': 'pupil response in percent change'})
#     ax[1].figure.axes[-1].yaxis.label.set_size(9)
#     ax[0].axvline(x=4000, linewidth=2, ls='--', color='w')
#     ax[1].axvline(x=5000, linewidth=2, ls='--', color='w')
#     ax[0].set_xticks(range(0, means_dphase.shape[1], 200))
#     xticklabels=np.arange((round(z[0])), round(z[-1]),0.2)
#     xticklabels=list(xticklabels)
#     xticklabels=[round(i,1) for i in xticklabels]
#     xticklabels=[str(i) for i in xticklabels]
#     ax[0].set_xticklabels(xticklabels)
#     ax[1].set_xticks(range(0, means_resp.shape[1], 200))
#     xticklabels=np.arange((round(x[0])), round(x[-1]),0.2)
#     xticklabels=list(xticklabels)
#     xticklabels=[round(i,1) for i in xticklabels]
#     xticklabels=[str(i) for i in xticklabels]
#     ax[1].set_xticklabels(xticklabels)
#     if os.getcwd() == 'C:\\Users\\Josefine\\Documents\\Promotion\\arousal_memory_experiments\\data\\pilot2_contrast_detection':
#         ax[0].set_yticklabels(['no stim','(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15'], rotation=0) #'(-10)-(-5)','(-5)-(-4.5)','(-4.5)-(-4)','(-4)-(-3.5)','(-3.5)-(-3)', '(-3)-(-2.5)','(-2.5)-(-2)','(-2)-(-1.5)','(-1.5)-(-1)', '(-1)-(-0.5)', '(-0.5)-0'      
#     else:
#         ax[0].set_yticklabels(['no stim','(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'], rotation=0) #'(-10)-(-5)','(-5)-(-4.5)','(-4.5)-(-4)','(-4)-(-3.5)','(-3.5)-(-3)', '(-3)-(-2.5)','(-2.5)-(-2)','(-2)-(-1.5)','(-1.5)-(-1)', '(-1)-(-0.5)', '(-0.5)-0'
#     ax[0].set_ylabel('SOA bin of auditory stimulus (in s; locked to decision phase onset)', fontsize=9) #locked to response
#     ax[1].set_ylabel('SOA bin of auditory stimulus (in s; locked to decision phase onset)', fontsize=9)
#     if os.getcwd() == 'C:\\Users\\Josefine\\Documents\\Promotion\\arousal_memory_experiments\\data\\pilot2_contrast_detection':
#         ax[1].set_yticklabels(['no stim','(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15'], rotation=0) #'(-10)-(-5)','(-5)-(-4.5)','(-4.5)-(-4)','(-4)-(-3.5)','(-3.5)-(-3)', '(-3)-(-2.5)','(-2.5)-(-2)','(-2)-(-1.5)','(-1.5)-(-1)', '(-1)-(-0.5)', '(-0.5)-0'      
#     else:
#         ax[1].set_yticklabels(['no stim','(-3)-(-2.65)','(-2.65)-(-2.3)','(-2.3)-(-1.95)','(-1.95)-(-1.6)','(-1.6)-(-1.25)', '(-1.25)-(-0.9)','(-0.9)-(-0.55)','(-0.55)-(-0.2)','(-0.2)-0.15', '0.15-0.5'], rotation=0) #'(-10)-(-5)','(-5)-(-4.5)','(-4.5)-(-4)','(-4)-(-3.5)','(-3.5)-(-3)', '(-3)-(-2.5)','(-2.5)-(-2)','(-2)-(-1.5)','(-1.5)-(-1)', '(-1)-(-0.5)', '(-0.5)-0'
#     ax[0].set_xlabel('time (in s) with respect to decision phase onset', fontsize=9)
#     ax[1].set_xlabel('time (in s) with respect to response', fontsize=9)

#     return fig

# # Heat maps for all subs by bin for pupil resp locked to AS
# def plot_pupil_bin_singles(epochs_p_stim, y, groupby=['block_id', 'trial_nr', 'soa_bin']):
    
#     fig = plt.figure(figsize=(16,10.5))
#     fig.suptitle('Pupil response by SOA of auditory stimulus (locked to decision phase onset)')
#     for sub_id, new_df in epochs_p_stim.groupby('subject_id'):

#         means_stim = new_df.groupby(groupby).mean().groupby(['soa_bin']).mean()
#         ax=fig.add_subplot(7,6,sub_id)
#         sns.heatmap(means_stim, ax=ax, vmin=-10, vmax=10, cbar_kws={'label': 'psc pupil response'})
#         #ax.figure.axes[-1].yaxis.label.set_size(9)
#         ax.axvline(x=2000, linewidth=0.7, ls='--', color='w')
#         ax.set_xticks(range(0, means_stim.shape[1], 1000))
#         xticklabels=np.arange((round(y[0])), round(y[-1]),1)
#         xticklabels=list(xticklabels)
#         xticklabels=[round(i,1) for i in xticklabels]
#         xticklabels=[str(i) for i in xticklabels]
#         ax.set_xticklabels(xticklabels)
#         ax.set_yticklabels(['','(-2.65)-(-2.3)','(-1.95)-(-1.6)', '(-1.25)-(-0.9)','(-0.55)-(-0.2)','0.15-0.5'], rotation=0) #'(-10)-(-5)', '(-4.5)-(-4)', '(-3.5)-(-3)', '(-2.5)-(-2)', '(-1.5)-(-1)', '(-0.5)-0'], rotation=0)
#         ax.set_ylabel('SOA bin of AS (in s)') # fontsize=9)
#         ax.set_xlabel('time (in s) with respect to AS') # , fontsize=9)

#     sns.despine(trim=True)
#     plt.tight_layout()

#     return fig

# def plot_repetition(df, binning='soa_bin', which_rep='repetition_past', look_back=5, trial_cutoff=11):

#     fig, ax= plt.subplots(1,1)
#     df = df.loc[(df['trial_nr']>trial_cutoff)]

#     df_plot = pd.DataFrame()

#     for j in range((look_back+1)):
#         df_plot = df_plot.append(pd.Series([df.groupby('subject_id').apply(lambda x: get_reps(x, i, j, which_rep)).mean() for i in sorted(df[binning].dropna().astype('int').unique())]), ignore_index=True)

#     for i in range(len(sorted(df[binning].dropna().astype('int').unique()))):
#         plt.plot(df_plot.iloc[:,i], color=sns.color_palette("viridis",12)[int(i)], ls='-', lw=1.5, label=i)

#     ax.set_xlabel('AS x trials ago', fontsize=8)
#     ax.set_ylabel('{}'.format(which_rep), fontsize=8)
#     plt.legend(bbox_to_anchor=(1,1), loc="upper left")

#     sns.despine(trim=True)
#     plt.tight_layout()

#     return fig, df_plot

# def get_reps(df, bin, shifting, which_rep, binning=binning):
#     rep_p = df.loc[(df[binning].shift(shifting)==bin), which_rep].mean()
#     return rep_p

# def plot_pupil_time(epochs_p_stim, groupby=['subject_id', 'condition'], ylim=None):

#     fig = plt.figure(figsize=(3.5,3.5))
#     ax = fig.add_subplot(221)
#     epochs_p_stim1 = epochs_p_stim.groupby(['trial_nr']).mean().iloc[0:99]
#     means1 = epochs_p_stim1.groupby(groupby).mean().groupby(['condition']).mean() #!work here
#     sems1 = epochs_p_stim1.groupby(groupby).mean().groupby(['condition']).sem()
#     x = np.array(means1.columns, dtype=float)
#     plt.fill_between(x, means1.iloc[0]-sems1.iloc[0], means1.iloc[0]+sems1.iloc[0], color=sns.color_palette()[1], alpha=0.2)
#     plt.plot(x, means1.iloc[0], color=sns.color_palette()[1],    ls='-', label='boost')
    
#     plt.axvline(0, color='k', ls='--')
#     plt.legend(loc=4)
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel('Time from stimulus (s)')
#     plt.ylabel('Pupil response (% change)')
        
#     fig = plt.figure(figsize=(3.5,3.5))
#     ax = fig.add_subplot(222)
#     epochs_p_stim2 = epochs_p_stim.groupby(['trial_nr']).mean().iloc[100:199]
#     means2 = epochs_p_stim2.groupby(groupby).mean().groupby(['condition']).mean()
#     sems2 = epochs_p_stim2.groupby(groupby).mean().groupby(['condition']).sem()
#     x = np.array(means2.columns, dtype=float)
#     plt.fill_between(x, means2.iloc[0]-sems2.iloc[0], means2.iloc[0]+sems2.iloc[0], color=sns.color_palette()[1], alpha=0.2)
#     plt.plot(x, means2.iloc[0], color=sns.color_palette()[1],    ls='-', label='boost')
    
#     plt.axvline(0, color='k', ls='--')
#     plt.legend(loc=4)
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel('Time from stimulus (s)')
#     plt.ylabel('Pupil response (% change)')

#     fig = plt.figure(figsize=(3.5,3.5))
#     ax = fig.add_subplot(223)
#     epochs_p_stim3 = epochs_p_stim.groupby(['trial_nr']).mean().iloc[200:299]
#     means3 = epochs_p_stim3.groupby(groupby).mean().groupby(['condition']).mean()
#     sems3 = epochs_p_stim3.groupby(groupby).mean().groupby(['condition']).sem()
#     x = np.array(means3.columns, dtype=float)
#     plt.fill_between(x, means3.iloc[0]-sems3.iloc[0], means3.iloc[0]+sems3.iloc[0], color=sns.color_palette()[1], alpha=0.2)
#     plt.plot(x, means3.iloc[0], color=sns.color_palette()[1],    ls='-', label='boost')
    
#     plt.axvline(0, color='k', ls='--')
#     plt.legend(loc=4)
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel('Time from stimulus (s)')
#     plt.ylabel('Pupil response (% change)')

#     fig = plt.figure(figsize=(3.5,3.5))
#     ax = fig.add_subplot(224)
#     epochs_p_stim4 = epochs_p_stim.groupby(['trial_nr']).mean().iloc[300:]
#     means4 = epochs_p_stim4.groupby(groupby).mean().groupby(['condition']).mean()
#     sems4 = epochs_p_stim4.groupby(groupby).mean().groupby(['condition']).sem()
#     x = np.array(means4.columns, dtype=float)
#     plt.fill_between(x, means4.iloc[0]-sems4.iloc[0], means4.iloc[0]+sems4.iloc[0], color=sns.color_palette()[1], alpha=0.2)
#     plt.plot(x, means4.iloc[0], color=sns.color_palette()[1],    ls='-', label='boost')
    
#     plt.axvline(0, color='k', ls='--')
#     plt.legend(loc=4)
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel('Time from stimulus (s)')
#     plt.ylabel('Pupil response (% change)')
    
#     sns.despine(trim=True)
#     plt.tight_layout()
#     return fig

# # JW's load data function which I use if I wanna read in individual blocks to try out stuff
# def load_data_old(filename):

#     from pyedfread import edf

#     import utils
#     import preprocess_pupil

#     filename = '35_1_2023_05_31_18_16_08.edf' #100_1_2022_09_27_12_51_14.edf' #'/home{}' #not necessary within defining the function?

#     #extract subject and session
#     subj = os.path.basename(filename).split('_')[0]
#     ses = os.path.basename(filename).split('_')[1]

#     # load pupil data and meta data: #
#     df_meta_all = pd.read_csv('35_1_2023_05_31_18_16_08_events.tsv', sep='\t') #not usually needed, exists within following code
#     try:
#         samples, events, messages = edf.pread(filename, trial_marker=b'')
#         df_meta_all = pd.read_csv(glob.glob('data/{}_{}*_events.tsv'.format(subj, ses))[0], sep='\t') #finds all the pathnames matching a specified pattern according to the rules used by the Unix shell
#     except Exception as e:
#         print(subj, ses, 'loading data failed')
#         print(e)
#         return pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])

#     samples = samples.loc[samples['time']!=0,:].reset_index(drop=True) #subset rows with time != 0 #with .loc you can access specific values in the dataframe[rowX:rowY, columnX:columnY]
#     samples = samples.rename({'pa_left': 'pupil'}, axis=1) # rename pupil area left to just pupil
#     messages = messages.reset_index(drop=True)

#     # subset behav data to only response rows and add subject and block id:
#     df_meta = df_meta_all.loc[(df_meta_all['phase']==2)&(df_meta_all['event_type']=='response')].reset_index(drop=True) # only for trials with a response that was given in phase 2
#     df_meta = df_meta.loc[(df_meta['response']=='y')|(df_meta['response']=='m'),:].reset_index(drop=True) # only for trials with a response that was either y or m
#     #df_meta= df_meta_all.loc[(df_meta_all['phase']==1)].reset_index() # we are only interested in the decision phase meta data + do I also need to include drop=True or no (because I'm keeping all trials)?
#     df_meta['subject_id'] = subj
#     df_meta['block_id'] = ses

#     # remove doubles:
#     df_meta = df_meta.iloc[np.where(df_meta['trial_nr'].diff()!=0)[0]]
#     # remove baseline trials:
#     df_meta = df_meta.loc[(df_meta['trial_nr']!=0)&(df_meta['trial_nr']!=1)&(df_meta['trial_nr']!=102)&(df_meta['trial_nr']!=203)&(df_meta['trial_nr']!=304)&(df_meta['trial_nr']!=405)] #cut off baseline trials
#     df_meta=df_meta.reset_index(drop=True)

#     ## add timestamps from eyetracker messages (for the start of each phase, for the response onset) calculate rt: 
#     df_meta['time_phase_0'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-0'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000 
#     df_meta['time_phase_1'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-1'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000
#     df_meta['time_phase_2'] = np.array([messages.loc[messages['trialid ']=='start_type-stim_trial-{}_phase-2'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000
#     # make variable for actual auditory stimulus onset, list is a helper
#     df_meta['actual_stim_start2']= np.array([messages.loc[messages['trialid ']=='post_sound_trial_{}'.format(i), 'trialid_time'] for i in df_meta['trial_nr']]).ravel() / 1000 #stays sequential and rows are not (?) in line with trial nr in df meta
#     list_actual_stim_start=[]
#     for i in range(0,len(df_meta['actual_stim_start2'])):
#         try:
#             k=df_meta['actual_stim_start2'][i].values[0]
#             list_actual_stim_start.append(k)
#         except:
#             list_actual_stim_start.append(np.NaN)
#     df_meta['actual_stim_start']=list_actual_stim_start

#     # make dataframe (df_resp) containing all trialid messages that belong to a valid response (that was given in phase 2):
#     df_resp=pd.DataFrame(columns=['trial_nr','trial_id'])
#     for i in df_meta['trial_nr']: 
#         value='start_type-response_trial-'+str(i)+'_' #create string that belongs to trial
#         for j in messages['trialid ']:
#             if value in j and 'phase-2' in j and ('key-m' in j or 'key-y' in j): #check whether response was y or m and given in phase 2
#                 row={"trial_nr":i,"trial_id":j}
#                 df_resp = pd.concat([df_resp, pd.DataFrame([row])])
#     df_resp = df_resp.iloc[np.where(df_resp['trial_nr'].diff()!=0)[0]] # remove the 2nd of double-responses
#     df_meta['time_response'] = np.array([messages.loc[messages['trialid ']=='{}'.format(i), 'trialid_time']for i in df_resp['trial_id']]).ravel()/1000
    
#     df_meta['rt'] = df_meta['time_response'] - df_meta['time_phase_2'] #reaction time
#     #df_meta['stim_start'] = df_meta['time_phase_2'] + df_meta['soa']
#     # get auditory stimulus start and then soa from messages (eyetracking data)
#     df_meta['actual_soa'] = df_meta['actual_stim_start'] - df_meta['time_phase_2'] #soa as calculated with actual as onset
#     #df_meta.loc[(df_meta.soa > df_meta.rt), 'condition'] = 'no_as' #label trials in which the AS wasn't played because the subj reacted beforehand

#     # datetime
#     timestamps = filename.split('_')[2:]
#     timestamps[-1] = timestamps[-1].split('.')[0]
#     df_meta['start_time'] = datetime.datetime(*[int(t) for t in timestamps])
#     df_meta['morning'] = df_meta['start_time'].dt.time <= datetime.time(13,0,0)
    
#     # check if not too many omissions:
#     if df_meta.shape[0]<340:
#         print(subj)
#         print(ses)
#         return pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
    
#     # add bin number of soa to df_meta
#     df_meta['soa_bin']=np.NaN
#     bins=[-3.01, -2.65, -2.3, -1.95, -1.6, -1.25, -0.9, -0.55, -0.2, 0.15, 0.51] # added a little buffer for as that was played a little too early or a little too late #[-3, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6]
#     bin_names=[1,2,3,4,5,6,7,8,9,10]  #[1,2,3,4,5,6,7,8,9,10,11,12]
#     df_meta['soa_bin']= pd.cut(df_meta['actual_soa'], bins, labels=bin_names)
#     df_meta['soa_bin']=df_meta['soa_bin'].astype('float').fillna(0).astype('int')
    
#     # # count occurences of different bins
#     counts = df_meta.groupby('soa_bin')['soa_bin'].count()
#     print(subj, ses, 'Bin counts:', counts)

#     ## make variable indicating hit(=1), miss(=2), fa(=3) or cr(=4)
#     df_meta['sdt']=np.nan
#     if int(subj)<=20:
#         df_meta.loc[(df_meta['stimulus']=='present') & (df_meta['response']=='m'), 'sdt'] = 1
#         df_meta.loc[(df_meta['stimulus']=='present') & (df_meta['response']=='y'), 'sdt'] = 2
#         df_meta.loc[(df_meta['stimulus']=='absent') & (df_meta['response']=='m'), 'sdt'] = 3
#         df_meta.loc[(df_meta['stimulus']=='absent') & (df_meta['response']=='y'), 'sdt'] = 4
#     else:
#         df_meta.loc[(df_meta['stimulus']=='present') & (df_meta['response']=='y'), 'sdt'] = 1
#         df_meta.loc[(df_meta['stimulus']=='present') & (df_meta['response']=='m'), 'sdt'] = 2
#         df_meta.loc[(df_meta['stimulus']=='absent') & (df_meta['response']=='y'), 'sdt'] = 3
#         df_meta.loc[(df_meta['stimulus']=='absent') & (df_meta['response']=='m'), 'sdt'] = 4

#     # preprocess pupil data (refers to preprocess_pupil script):
#     df_bw = df_meta_all.loc[(df_meta_all['event_type']=='bw1')].reset_index(drop=True) # only for tasks with a response
#     df_bw['subject_id'] = subj
#     df_bw['block_id'] = ses
#     fs = int(1/samples['time'].diff().median()*1000) #what was fs again? frequency spectrum?
#     params = {'fs':fs, 'lp':10, 'hp':0.01, 'order':3}
#     df = preprocess_pupil.preprocess_pupil(samples=samples, events=events, params=params, df_bw=df_bw)
#     df['time'] = df['time'] / 1000

#     # plot result:
#     try:

#         columns = ['subject_id', 'block_id', 'trial_nr', 'condition', 'actual_soa'] #it's called condition here

#         # make epochs (from utils)
#         # locked to the start of the AS
#         epochs = utils.make_epochs(df=df, df_meta=df_meta, locking='actual_stim_start', start=-2, dur=7, measure='pupil_int_lp_psc', fs=fs,   #[df_meta['condition']=='boost'] #'actual_stim_start'
#                         baseline=False, b_start=-1, b_dur=1)
#         epochs[columns] = df_meta[columns]
#         epochs_p_stim = epochs.set_index(columns)

#         #locked to the response
#         epochs = utils.make_epochs(df=df, df_meta=df_meta, locking='time_response', start=-5, dur=7, measure='pupil_int_lp_psc', fs=fs, 
#                             baseline=False, b_start=-1, b_dur=1)
        
#         epochs[columns] = df_meta[columns]
#         epochs_p_resp = epochs.set_index(columns)

#         #locked to decisionphase onset:
#         epochs = utils.make_epochs(df=df, df_meta=df_meta, locking='time_phase_2', start=-4, dur=7, measure='pupil_int_lp_psc', fs=fs, 
#                             baseline=False, b_start=-1, b_dur=1)
        
#         epochs[columns] = df_meta[columns]
#         epochs_p_dphase = epochs.set_index(columns) #for onset decision phase

        
#         # epochs for blinks??
#         # epochs = make_epochs(df=df, df_meta=df_meta, locking='time_phase_2', start=-3, dur=6, measure='is_blink', fs=fs, 
#         #                 baseline=False, b_start=-1, b_dur=1)
#         # epochs[columns] = df_meta[columns]
#         # epochs_b_resp = epochs.set_index(columns)
        
#         # get blinks and saccades in df_meta explicitely
#         df_meta['blinks'] = [df.loc[(df['time']>(df_meta['time_phase_1'].iloc[i]-0))&(df['time']<df_meta['time_phase_2'].iloc[i]), 'is_blink_eyelink'].mean() 
#                     for i in range(df_meta.shape[0])]
#         df_meta['sacs'] = [df.loc[(df['time']>(df_meta['time_phase_1'].iloc[i]-0))&(df['time']<df_meta['time_phase_2'].iloc[i]), 'is_sac_eyelink'].mean() 
#                     for i in range(df_meta.shape[0])]


#         # fig = plt.figure(figsize=(12,6))
#         # ax = fig.add_subplot(211)
#         # plt.plot(samples['time'], samples['pupil'])
#         # plt.plot(samples['time'], samples['pupil_int'])
#         # plt.plot(samples['time'], samples['pupil_int_lp'])
#         # blinks = events.loc[events['blink']]
#         # for i in range(blinks.shape[0]):
#         #     plt.axvspan(blinks['start'].iloc[i], blinks['end'].iloc[i], color='r', alpha=0.2)

#         # ax = fig.add_subplot(212)
#         # plt.plot(samples['time'], samples['pupil_int_lp_psc'])
#         # fig.savefig('figs/sessions/{}_{}.jpg'.format(subj, ses))

#         blinks = events.loc[events['blink']]
#         blinks['start'] = blinks['start'] / 1000
#         blinks['end'] = blinks['end'] / 1000

#         import matplotlib.gridspec as gridspec
        
#         start = float(df['time'].iloc[0])
#         end = float(df['time'].iloc[-1])

#         # plot:
#         fig = plt.figure(figsize=(7.5, 2.5))
#         gs = gridspec.GridSpec(5, 1)

#         ax = fig.add_subplot(gs[0, :])
#         for t, d in df_meta.groupby(['trial_nr']):
#             ax.axvspan((float(d['time_phase_1'])-start)/60, (float(d['time_phase_2'])-start)/60, color='grey', lw=0)
#         plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#         plt.xlim(0, (end-start)/60)
#         plt.ylim(0,2)

#         ax = fig.add_subplot(gs[1, :])
#         for t, d in df_meta.loc[df_meta['stimulus']=='noise_500',:].groupby(['trial_nr']): # could also contain noise_100 or noise_200
#             ax.axvspan((float(d['time_phase_1'])-1-start)/60, (float(d['time_phase_2'])-start)/60, color='red', lw=0)
#         plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#         plt.xlim(0, (end-start)/60)
#         plt.ylim(0,2)

#         ax = fig.add_subplot(gs[2:5, :])
#         for i in range(blinks.shape[0]):
#             ax.axvspan((blinks['start'].iloc[i]-start)/60, (blinks['end'].iloc[i]-start)/60, color='red', alpha=0.2, lw=0)
#         ax.plot((df['time'].iloc[::10]-start)/60, df['pupil_int_lp_psc'].iloc[::10])
#         plt.xlim(0, (end-start)/60)
#         plt.xlabel('Time (min)')

#         sns.despine(trim=False)
#         plt.tight_layout()

#         fig.savefig('figs/sessions/{}_{}.pdf'.format(subj, ses))
        
#         # plot evoked pupil responses (i)locked to stim (ii) locked to resp by cond (iii) locked to decision phase onset by cond:
#         x = np.array(epochs_p_resp.columns, dtype=float)
#         y = np.array(epochs_p_stim.columns, dtype=float)
#         z = np.array(epochs_p_dphase.columns, dtype=float)

#         baselines = np.atleast_2d(epochs_p_resp.loc[:,(x>-4)&(x<-3)].mean(axis=1)).T
#         baselines2 = np.atleast_2d(epochs_p_stim.loc[:,(y>-0.5)&(y<0)].mean(axis=1)).T
#         baselines3 = np.atleast_2d(epochs_p_dphase.loc[:,(z>-3.5)&(z<-3)].mean(axis=1)).T

#         # use trial-individual baselines:
#         epochs_p_resp_b = epochs_p_resp - baselines3 #or use some version of baselines
#         epochs_p_stim_b = epochs_p_stim - baselines2
#         epochs_p_dphase_b = epochs_p_dphase - baselines3

#         # deduce mean of trial-individual baselines:
#         # epochs_p_resp_b = epochs_p_resp - np.mean(baselines)
#         # epochs_p_stim_b = epochs_p_stim - np.nanmean(baselines2)
#         # epochs_p_dphase_b = epochs_p_dphase - np.mean(baselines3)


#         # binning?:
#         # ind = np.array(df_meta['sound_onset'] > -3 & df_meta['sound_onset']<=-2) + np.array(df_meta['condition']=='normal') #for soa perceptual experiment

#         #fig = plot_pupil_responses(epochs_p_stim_b.loc[ind,:].iloc[:,::10], #for soa perceptual experiment
#         #                            epochs_p_resp_b.loc[ind,:].iloc[:,::10], 
#         #                            groupby=['trial_nr', 'condition'])

#         fig = plot_pupil_responses(epochs_p_stim_b.iloc[:,::10], #as onset
#                                     epochs_p_resp_b.iloc[:,::10], #response onset
#                                     epochs_p_dphase_b.iloc[:,::10], #decision phase onset locked
#                                     groupby=['trial_nr', 'condition']) 
#         fig.savefig('figs/{}_{}_2.pdf'.format(subj, ses))

#         #means_epochs = epochs_p_stim_b.groupby('stimulus').mean() #for plotting means of each type of stimulus seperately
#         #plt.plot(means_epochs.iloc[0])

#         #plot pupil response over time (not working yet)
#         fig = plot_pupil_time(epochs_p_stim_b.iloc[:,::10], #as onset
#                                     groupby=['trial_nr', 'condition'])



#     except Exception as e:
#         print(subj, ses)
#         print(e)
#         return pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])

#     return df_meta, epochs_p_stim, epochs_p_resp


