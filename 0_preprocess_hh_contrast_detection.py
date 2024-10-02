'''
First script to be applied to raw data of Experiment 2 which took place in Hamburg and was a simple contrast detection experiment. 
Handles preprocessing.

Takes following inputs:
- edf files with eye data
- tsv files with behavioral data

Outputs:
- df_meta.csv: concatenated dataframe with all behavioral data
- epochs... ...hdf: concatenated epochs of eye data (1 row per trial) locked to respective events (dphase = decisionphase, stim = task-irrelevant stimulus, resp = button press)
'''

# imports
import os, glob, datetime
from functools import reduce
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from joblib import Parallel, delayed
from tqdm import tqdm

import utils_hh_contrast_detection

# define in- and output folders
project_dir = 'C:\\Users\\Josefine\\Documents\\Promotion\\pilot_study\\arousal_percept_experiments' # os.path.dirname(os.getcwd()) # parent directory
data_dir = os.path.join(project_dir, 'data', 'contrast_detection')  
figs_dir = os.path.join(data_dir, 'figs') # os.path.join(project_dir, 'figs')
n_jobs = 8 
os.chdir(project_dir)
data_set = 'hh_contrast_detection'

# define list of eyetracking files to be loaded (behavioral data is loaded at the same time)
edf_filenames = glob.glob(os.path.join(data_dir, '*.edf'))
print(edf_filenames)

# apply load_data function that reads in behavioral and eye data, preprocesses and epochs it
res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(utils_hh_contrast_detection.load_data)(filename) #, figs_dir) 
                                                                for filename in tqdm(edf_filenames))

# unpack: 
df_meta = pd.concat([res[i][0] for i in range(len(res))]).reset_index(drop=True) # behavior
epochs_p_sound = pd.concat([res[i][1] for i in range(len(res))]) # locked to task-irrelevant white noise stimulus (TIS; used to be called epochs_p_stim)
epochs_p_resp = pd.concat([res[i][2] for i in range(len(res))]) # locked to decision response (button press)
epochs_p_dphase = pd.concat([res[i][3] for i in range(len(res))]) # locked to onset of decision phase
# epochs_p_bl10 = pd.concat([res[i][4] for i in range(len(res))]) # 10 s break intervals (5 per block)
# epochs_cp_bl10 = pd.concat([res[i][5] for i in range(len(res))]) # see above but cp means custom pupil range which uses the black and white screen pupil sizes as reference points for 0 and 100 % pupil values
# epochs_cp_stim = pd.concat([res[i][6] for i in range(len(res))])
# epochs_cp_resp = pd.concat([res[i][7] for i in range(len(res))])
# epochs_cp_dphase = pd.concat([res[i][8] for i in range(len(res))])


# add some variables and change values for clarity or to make more convenient for statistics
df_meta['condition'] = (df_meta['condition']=='boost').astype(int)
df_meta['stimulus'] = df_meta['stimulus'].map({'absent': 0, 'present': 1}).astype(int)
df_meta['choice'] = 0
df_meta.loc[(df_meta['stimulus']==1)&(df_meta['correct']==1), 'choice'] = 1
df_meta.loc[(df_meta['stimulus']==0)&(df_meta['correct']==0), 'choice'] = 1
df_meta['block_id'] = df_meta['block_id'].astype(int)-1
df_meta['date'] = pd.to_datetime(df_meta['start_time']).dt.date
df_meta['block_split'] = (df_meta['trial_nr']>=204).map({False: 1, True: 2}) # FIXME 
df_meta.loc[df_meta['trial_nr']<=11, 'block_split'] = 0

# add total number of trials:
df_meta['nr_trials_total'] = df_meta.groupby(['subject_id'])['trial_nr'].transform('count')
print(df_meta.groupby(['subject_id'])['nr_trials_total'].mean())

# add accuracy per block:
df_meta = df_meta.groupby(['subject_id', 'block_id'], group_keys=False).apply(utils_hh_contrast_detection.compute_accuracy, trial_cutoff=11)
print(df_meta.groupby(['subject_id', 'block_id'])['correct_avg'].mean())

# exclude trials that were affected by noise of construction work
loud_trials = pd.DataFrame([[9,1,6],[9,1,7],[9,1,8],[11,3,354],[11,3,355],[11,3,356],[20,2,23],[20,2,54],[20,2,55],[20,2,56],[20,2,57],[20,2,58],[20,2,205],[20,2,301],[20,2,302],
                            [20,3,45],[20,3,46],[20,3,136],[20,3,137],[21,1,83],[21,1,89],[23,1,22],[23,1,23],[23,1,24],[23,1,25],[23,1,26],[23,1,27],[23,1,28],[23,1,29],[23,1,30],
                            [23,1,142],[23,1,151],[23,1,152],[23,1,153],[23,1,154],[23,1,155],[23,1,156],[23,1,157],[23,1,158],[23,1,159],[23,1,160],[23,1,161],[23,1,162],[23,1,163],
                            [23,1,164],[23,1,165],[23,1,166],[23,1,167],[23,1,188],[23,1,189],[23,1,190],[23,1,191],[23,1,192],[23,1,193],[23,1,194],[23,1,195],[23,1,196],[23,1,197],
                            [23,1,198],[23,1,199],[23,1,200],[23,1,201],[23,1,202],[23,1,203],[23,1,204],[23,1,205],[23,1,206],[23,1,298],[23,1,299],[23,1,300],[23,1,301],[23,1,302],
                            [23,1,303],[23,1,304],[23,1,305],[23,1,306],[23,1,307],[24,2,219],[24,2,220],[24,2,221],[24,2,222],[24,2,223],[24,2,224],[24,2,225],[24,2,226],[24,2,227],
                            [24,2,228],[24,2,229],[24,2,230],[24,2,231],[24,2,232],[24,2,233],[24,2,234],[24,2,235],[24,2,236],[24,2,237],[24,2,238],[24,2,239],[24,2,240],[24,2,241],
                            [24,2,242],[24,2,243],[24,2,244],[24,2,245],[24,2,246],[24,2,247],[24,2,248],[24,2,249],[24,2,250],[24,2,251],[24,2,252],[24,2,253],[24,2,254],[24,2,255]],
                            columns=['subject_id', 'block_id','trial_nr'])
for idx, row in loud_trials.iterrows(): 
    df_meta.drop(df_meta.loc[(df_meta['subject_id']==row['subject_id']) & (df_meta['block_id']==row['block_id']) & (df_meta['trial_nr']==row['trial_nr'])].index, inplace=True)
    epochs_p_dphase.drop(epochs_p_dphase.loc[(epochs_p_dphase.index.get_level_values(0)==row['subject_id']) & (epochs_p_dphase.index.get_level_values(1)==row['block_id']) & (epochs_p_dphase.index.get_level_values(2)==row['trial_nr'])].index, inplace=True)
    epochs_p_resp.drop(epochs_p_resp.loc[(epochs_p_resp.index.get_level_values(0)==row['subject_id']) & (epochs_p_resp.index.get_level_values(1)==row['block_id']) & (epochs_p_resp.index.get_level_values(2)==row['trial_nr'])].index, inplace=True)
    epochs_p_sound.drop(epochs_p_sound.loc[(epochs_p_sound.index.get_level_values(0)==row['subject_id']) & (epochs_p_sound.index.get_level_values(1)==row['block_id']) & (epochs_p_sound.index.get_level_values(2)==row['trial_nr'])].index, inplace=True)


# save:
print('saving... {}'.format(data_set))
df_meta.to_csv(os.path.join(data_dir, '{}_df_meta.csv'.format(data_set)))
epochs_p_sound.to_hdf(os.path.join(data_dir, '{}_epochs_p_sound.hdf'.format(data_set)), key='pupil')
epochs_p_resp.to_hdf(os.path.join(data_dir, '{}_epochs_p_resp.hdf'.format(data_set)), key='pupil')
epochs_p_dphase.to_hdf(os.path.join(data_dir, '{}_epochs_p_dphase.hdf'.format(data_set)), key='pupil')
# epochs_p_bl10.to_hdf(os.path.join(data_dir, '{}_epochs_p_bl10.hdf'.format(data_set)), key='pupil')
# epochs_cp_bl10.to_hdf(os.path.join(data_dir, '{}_epochs_cp_bl10.hdf'.format(data_set)), key='pupil')
# epochs_cp_stim.to_hdf(os.path.join(data_dir, '{}_epochs_cp_stim.hdf'.format(data_set)), key='pupil')
# epochs_cp_resp.to_hdf(os.path.join(data_dir, '{}_epochs_cp_resp.hdf'.format(data_set)), key='pupil')
# epochs_cp_dphase.to_hdf(os.path.join(data_dir, '{}_epochs_cp_dphase.hdf'.format(data_set)), key='pupil')