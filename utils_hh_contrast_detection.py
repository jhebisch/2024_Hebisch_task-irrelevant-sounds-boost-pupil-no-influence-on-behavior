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
