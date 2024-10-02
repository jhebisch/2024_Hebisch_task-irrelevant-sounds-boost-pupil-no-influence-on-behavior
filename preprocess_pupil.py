'''
author: JW de Gee
comment: Josefine Hebisch

preprocess_pupil() is the function that can be called from this script to do the complete preprocessing. all other functions are used within preprocess_pupil().

Following input is needed:
+ samples and events dataframes from edf file
+ params:
    * 'fs': frequency per second with which the pupil size was recorded (most cases 1000)
    * 'lp': low pass filter cut-off
    * 'hp': high pass filter cut-off
    * 'order': order that should be used for the butterworth filters (low and high)

Output:
Samples dataframe that includes pupil size measure 'pupil_int_lp_clean' that:
    * has interpolated blinks (as detected by eyelink & as detected by custom function)
    * was lowpassfiltered
    * has been cleaned by regressing out pupil responses to blinks and saccades (Knapen et al., 2016)
    * was converted to percent change using the median

There are also options for:
+ high pass filtering the low pass filtered measure, thus creating a bandpass
+ using fraction and slope functions
+ detecting blinks (that weren't detected by eyelink) (JW is not quite happy with it so he hasn't been using it)

'''

import os, glob
import numpy as np
import scipy as sp
from scipy import stats
from scipy import signal
import pandas as pd
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns

import make_epochs

from IPython import embed as shell

def _double_gamma(params, x):
    a1 = params['a1']
    sh1 = params['sh1']
    sc1 = params['sc1']
    a2 = params['a2']
    sh2 = params['sh2']
    sc2 = params['sc2']
    return a1 * sp.stats.gamma.pdf(x, sh1, loc=0.0, scale = sc1) + a2 * sp.stats.gamma.pdf(x, sh2, loc=0.0, scale=sc2)

def _butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = sp.signal.butter(order, [high], btype='lowpass')
    return b, a

def _butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = _butter_lowpass(highcut, fs, order=order)
    y = sp.signal.filtfilt(b, a, data)
    return y

def _butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = sp.signal.butter(order, [low], btype='highpass')
    return b, a

def _butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = _butter_highpass(lowcut, fs, order=order)
    y = sp.signal.filtfilt(b, a, data)
    return y

# def _detect_blinks(df, fs, measure='pupil', coalesce_period=1):

#     blink_periods = []
#     ts = np.array(df[measure])
#     ts = np.concatenate(( np.array([0]), np.diff(ts) )) # / (1/fs) # first derivative
    
#     k = 25
#     q1 = np.percentile(ts,25)
#     q3 = np.percentile(ts,75)
#     iqr = q3 - q1
#     cutoff_low = q1 - (k*iqr)
#     cutoff_high = q3 + (k*iqr)
#     if np.sum(ts < -cutoff_low) > 0:
#         blink_periods.append(np.array(df.loc[np.where((ts < cutoff_low))[0], 'time']))
#     if np.sum(ts > cutoff_high) > 0:
#         blink_periods.append(np.array(df.loc[np.where((ts > cutoff_high))[0], 'time']))

#     if len(blink_periods) == 0:
#         blink_periods = np.array([])
#     else:
#         blink_periods = np.sort(np.concatenate(blink_periods).ravel())

#     # additionally use eyelink blink timestamps:
#     blink_periods = np.sort(np.concatenate((blink_periods, np.array(df.loc[np.where((df['is_blink_eyelink'] == 1))[0], 'time']))))

#     # don't start and end with blink:
#     blink_periods = blink_periods[blink_periods != df['time'].iloc[0]]
#     blink_periods = blink_periods[blink_periods != df['time'].iloc[-1]]
    
#     # return:
#     if len(blink_periods) > 0:
#         blink_start_ind = np.where(np.concatenate((np.array([True]), np.diff(blink_periods) > coalesce_period)))[0]
#         blink_end_ind = np.concatenate(((blink_start_ind - 1)[1:], np.array([blink_periods.shape[0] - 1])))
#         return blink_periods[blink_start_ind], blink_periods[blink_end_ind]
#     else:
#         return None, None

def interpolate_blinks(df, measure, blink_column, fs, buffer=0.1):

    df['{}_int'.format(measure)] = df[measure].copy() #df = samples, measure=pupil, 
    starts = np.where(df[blink_column].diff()==1)[0] #blink_column=is_blink_eyelink
    ends = np.where(df[blink_column].diff()==-1)[0]  #get start and end indeces
    for s, e in zip(starts, ends):
        df.loc[s:e, 'is_blink'] = np.ones(e-s+1) #make new is_blink column (a copy of the other one?)
        s -= int(buffer*fs)
        e += int(buffer*fs)
        s = max((0,s)) #if s<0, take 0
        e = min((df.shape[0]-1,e)) #if e>end take end
        df.loc[s:e,'{}_int'.format(measure)] = np.linspace(df.loc[s,'{}_int'.format(measure)],
                                                            df.loc[e,'{}_int'.format(measure)], e-s+1)

def regress_blinks(df, interval=7, regress_blinks=True, regress_sacs=True):

    ''' 
    This function results from Knapen et al. (2016). There, pupil responses to blinks were extracted 
    from the pupil signal using least squares deconvolution and fitting a (double (for blinks)) gamma density functions. So here, 
    a gamma density function is created with the estimates from that paper which is then used as a kernel to convolve
    with a matrix in which the time points of blink ends and saccade ends are described. The result is used as a regressor
    to be applied to the pupil data of the according times.

    Alternatively, it would also be possible to estimate the pupil response in the current data set first and then use the resulting values in this function.
    '''
    
    fs = 1000
    blink_column='is_blink_new'
    sac_column='is_sac_eyelink'

    # only regress out blinks within these limits:
    early_cutoff = 25
    late_cutoff = interval

    # params:
    x = np.linspace(0, interval, int(interval * fs), endpoint=False)
    standard_blink_parameters = {'a1':-0.604, 'sh1':8.337, 'sc1':0.115, 'a2':0.419, 'sh2':15.433, 'sc2':0.178}
    blink_kernel = _double_gamma(standard_blink_parameters, x)
    standard_sac_parameters = {'a1':-0.175, 'sh1': 6.451, 'sc1':0.178, 'a2':0.0, 'sh2': 1, 'sc2': 1}
    sac_kernel = _double_gamma(standard_sac_parameters, x)

    # create blink regressor:
    blink_ends = np.where(df[blink_column].diff()==-1)[0] # in samples
    blink_ends = blink_ends[(blink_ends > (early_cutoff*fs)) & (blink_ends < (df.shape[0]-(late_cutoff*fs)))]
    if blink_ends.size == 0:
        blink_ends = np.array([0], dtype=int)
    else:
        blink_ends = blink_ends.astype(int)
    blink_reg = np.zeros(df.shape[0])
    blink_reg[blink_ends] = 1
    blink_reg_conv = sp.signal.fftconvolve(blink_reg, blink_kernel, 'full')[:-(len(blink_kernel)-1)] #fftconvolve uses fast fourier transformation for a fast convolution

    # create saccade regressor:
    sac_ends = np.where(df[sac_column].diff()==-1)[0] # in samples
    sac_ends = sac_ends[(sac_ends > (early_cutoff*fs)) & (sac_ends < (df.shape[0]-(late_cutoff*fs)))]
    if sac_ends.size == 0:
        sac_ends = np.array([0], dtype=int)
    else:
        sac_ends = sac_ends.astype(int)
    sac_reg = np.zeros(df.shape[0])
    sac_reg[sac_ends] = 1
    sac_reg_conv = sp.signal.fftconvolve(sac_reg, sac_kernel, 'full')[:-(len(sac_kernel)-1)]

    # combine regressors:
    regs = []
    regs_titles = []
    if regress_blinks:
        regs.append(blink_reg_conv)
        regs_titles.append('blink')
    if regress_sacs:
        regs.append(sac_reg_conv)
        regs_titles.append('saccade')
    print([r.shape for r in regs])

    # GLM:
    design_matrix = np.matrix(np.vstack([reg for reg in regs])).T
    betas = np.array(((design_matrix.T * design_matrix).I * design_matrix.T) * np.matrix(df['pupil_int_bp'].values).T).ravel()
    explained = np.sum(np.vstack([betas[i]*regs[i] for i in range(len(betas))]), axis=0)
    rsq = sp.stats.pearsonr(df['pupil_int_bp'].values, explained)[0]**2
    print('explained variance = {}%'.format(round(rsq*100,2)))
    
    # cleaned-up time series:
    df['pupil_int_bp_clean'] = df['pupil_int_bp'] - explained
    df['pupil_int_lp_clean'] = df['pupil_int_bp_clean'] + (df['pupil_int_lp']-df['pupil_int_bp'])

def temporal_filter(df, measure, fs=15, hp=0.01, lp=6.0, order=3):
    df['{}_lp'.format(measure)] = _butter_lowpass_filter(data=df[measure], highcut=lp, fs=fs, order=order)
    df['{}_bp'.format(measure)] = _butter_highpass_filter(data=df[measure], lowcut=hp, fs=fs, order=order) - (df[measure] - df['{}_lp'.format(measure)])

def psc(df, measure):
    df['{}_psc'.format(measure)] = (df[measure] - df[measure].median()) / df[measure].median() * 100

def fraction(df, measure):
    df['{}_frac'.format(measure)] = df[measure] / np.percentile(df[measure], 99.5)

def slope(df, measure, hp=2.0, fs=15, order=3):
    slope = np.concatenate((np.array([0]), np.diff(df[measure]))) * fs
    slope = _butter_lowpass_filter(slope, highcut=hp, fs=fs, order=order)
    df['{}_slope'.format(measure)] = slope

def missed_blink_detection(samples,delta=25):
    '''
    the following blink detection is courtesy of Rudy van den Brink and 
    can be found here https://github.com/rudyvdbrink/Pupil_code/ 
    '''
    # define interpolation settings
    ninterps = 100 #the number of iterations for the interpolation algorithm
    delta = delta  #the slope (in pixels) that we consider to be artifactually large  
    #(default value is 25) both the above variables would need to be changed if the sample rate is
    #not 1000 Hz, or if the units of the pupil aren't in pixels. Also, it's
    #best to tailor this for each participant so that all artifacts are
    #accuratlely identified. 

    #first,find bad data sections by slope and set them to zero. make
    #multiple passes through the data to find peaks that occur over
    #multiple time-points. note that 'ninterps' assumes a sampling rate
    #of 1000Hz, modify if the sampling rate differs (e.g. at 60 Hz, the
    #equivalent would be 6 iterations).
    y = samples['pupil'].copy() #pupil size
    pointi = 0 

    for i in np.linspace(0,ninterps):
    #find points with an artifactual derivative
        while pointi < len(y)-2:
            pointi+=1
            if np.abs(y[pointi] - y[pointi+1]) > delta:
                y[pointi] = 0
                   
    #in case the very end of the recording is bad data, we cannot
    #interpolate across it, so we manually set it to the mean of the recording
    points = np.array(np.where(y == 0)).ravel()
    if y[len(y)-1] == 0:
        samples['pupil'][(len(samples['pupil'])-1)-ninterps:(len(samples['pupil'])-1)] = np.nanmean((samples['pupil'].loc[samples['pupil']!=0]))
        points=np.delete(points,-1)  

    #in case the beginning of the recording is bad data, we cannot
    #interpolate across it, so we manually set it to the mean of the recording
    if y[0] == 0:
        samples['pupil'][:ninterps] = np.nanmean((samples['pupil'].loc[samples['pupil']!=0]))
        points=np.delete(points,0)

    #set all found indices to indicate a blink 
    #if it wasn't detected as one before
    
    print('eyelink blinks: ', samples['is_blink_eyelink'].sum(), 'blink locations: ', points)   
    for j in points:
        if samples['is_blink_new'][j]==0:
            samples['is_blink_new'][j]=1
    print('eyelink + missed blinks: ', samples['is_blink_new'].sum()) #show how many blinks have been added

    return samples

def custom_percent(samples, df_bw, measure, fs):
    #make custom pupil range for each participant based on black & white screen and convert sizes to according percentages

    # make black/white screen epochs based on the pre-processed df int_lp pupil
    base_columns = ['subject_id', 'block_id', 'trial_nr']
    bw_epochs = make_epochs.make_epochs(df=samples, df_meta=df_bw, locking='onset', start=2, dur=16, measure=measure, fs=fs,   #[df_meta['condition']=='boost']
        baseline=False)
    bw_epochs[base_columns] = df_bw[base_columns]
    epochs_p_bw = bw_epochs.set_index(base_columns)

    # find min and max pupil size within black and white screen interval
    pupil_0 = epochs_p_bw[0:1].max().min() #it is assumed that the lowest size happens during the white screen
    pupil_100 = epochs_p_bw[0:1].max().max()-pupil_0 #it is assumed that the highest size happens during the black screen

    #set custom pupil range
    samples['{}_cp'.format(measure)] = (samples[measure] - pupil_0) / pupil_100 * 100
    return samples

# here is where it all actually happens
def preprocess_pupil(samples, events, params, df_bw=None):

    # add blink indices to samples (0 means no blink, 1 means blink):
    samples['is_blink_eyelink'] = 0
    for start, end in zip(events.loc[events['blink']==1, 'start'], events.loc[events['blink']==1, 'end']):
        samples.loc[(samples['time']>=start)&(samples['time']<=end), 'is_blink_eyelink'] = 1

    # add saccade indices:
    samples['is_sac_eyelink'] = 0
    for start, end in zip(events.loc[events['type']=='saccade', 'start'], events.loc[events['type']=='saccade', 'end']):
        samples.loc[(samples['time']>=start)&(samples['time']<=end), 'is_sac_eyelink'] = 1
    
    # set is_sac_eyelink to 0, when is_blink_eyelink:
    samples.loc[samples['is_blink_eyelink']==1, 'is_sac_eyelink'] = 0
    
    # find blinks that eyelink missed
    samples['is_blink_new']=samples['is_blink_eyelink'].copy()
    missed_blink_detection(samples,delta=25)
    # set is_sac_eyelink to 0, when is_blink_eyelink:
    samples.loc[samples['is_blink_new']==1, 'is_sac_eyelink'] = 0

    # blink interpolate:
    samples['is_blink'] = 0
    interpolate_blinks(df=samples, measure='pupil', blink_column='is_blink_new', fs=params['fs'], buffer=0.2)

    # blink interpolate step 2:
    # blink_starts, blink_ends = _detect_blinks(df, fs, blink_detection_measures, coalesce_period=coalesce_period)
    # interpolate_blinks(df=df, fs=params['fs'], measure='pupil', blink_detection_measures=['pupil'],)

    # temporal filter
    temporal_filter(df=samples, measure='pupil_int', fs=params['fs'], hp=params['hp'], lp=params['lp'], order=params['order'])

    # regress out pupil responses to blinks and saccades:
    regress_blinks(df=samples, interval=7, regress_blinks=True, regress_sacs=True)

    # percent signal change:
    psc(df=samples, measure='pupil_int_lp_clean')
    psc(df=samples, measure='pupil_int_lp')

    # custom percent pupil size
    if df_bw is not None:
        custom_percent(samples=samples, df_bw=df_bw, measure='pupil_int_lp_clean', fs=params['fs'])
        custom_percent(samples=samples, df_bw=df_bw, measure='pupil_int_lp', fs=params['fs'])

    # fraction(df=df, measure='pupil_int_lp')
    # slope(df=df, measure='pupil_int_lp_frac', hp=params['s_hp'], fs=params['fs'], order=params['order'])    
    return samples
