#%%
import mne
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import meegkit
from mne.datasets import sample
from mne.stats.regression import linear_regression_raw
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import scipy
#%% time-resolved regression to get regressors for lexical variables for (overlapping) fixation related potentials

#%% paths
dir_raw = '/Volumes/Blue1TB/EyeMindLink/Data'
dir_fif = '/Volumes/Blue1TB/EEG_processed/preprocessed_fif/'
event_fn_suffix = '_eyetracker_events.csv' # inside dir_fif, events containing fixations sacs and blinks as well as tasks, generated by merge_eyetracker_eeg.py
dir_out = '/Volumes/Blue1TB/EEG_processed/FRP_TRF_lexical/'
os.makedirs(dir_out, exist_ok=True)
dir_events = os.path.expanduser('~/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/Processed/events/') # task events
ia_df = pd.read_csv('../info/ia_label_mapping_opt_surprisal.csv').rename(columns={'opt-125m_surprisal_wholetext':'surprisal', 'opt-125m_surprisal_page':'surprisal_page'})
ia_df['IA_ID'] = ia_df['IA_ID'].fillna('-1').astype(float).astype(int).astype(str)
# set to NA any surprisal or frerquency for stop words and punctuation
ia_df['word_freq'] = ia_df['word_freq'].where(~ia_df['stop_word'] & ~ia_df['punctuation'])
ia_df['surprisal'] = ia_df['surprisal'].where(~ia_df['stop_word'] & ~ia_df['punctuation'])

beh_df = pd.read_csv('~/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/Processed/Behaviour/EML1_page_level.csv') # comp and MW scores
eeg_trigger_df = pd.read_csv('../info/EEGtriggerSources.csv')

# %% test subj
pID = 'EML1_028'
# %% load 
EEG = mne.io.read_raw_fif(os.path.join(dir_fif, f'{pID}_p.fif'), preload=True)
events = pd.read_csv(os.path.join(dir_fif, f'{pID}{event_fn_suffix}'))
# treat '.' as missing in 'IA_ID'
events['IA_ID'] = events['IA_ID'].replace('.', np.nan)
events['IA_ID'] = events['IA_ID'].fillna('-1')
events['eeg_sample'] = events['eeg_sample'].astype(float).astype(int)
# replace Fixation_R reparsed with Fixation_R
events['event_type'] = events['event_type'].replace('Fixation_R-reparsed','Fixation_R')
events['identifier'] = events['identifier'].fillna('') # some are NaN
events['task'] = events['task'].fillna('none') # some are NaN
events['event_type'] = events['event_type'].fillna('other') # some are NaN
events = events.drop_duplicates(subset=['latency_sec','description'])
beh_df_i = beh_df[beh_df['ParticipantID']==pID]
beh_df_i['identifier'] = beh_df_i['Text'].astype(str) + (beh_df_i['PageNum']-1).astype(str)
# merge lexical properties to events by IA_ID, which needs to be forced to be a string formatted as an integer not a float like 1.0
events = events.merge(ia_df, how='left')
events = events.merge(beh_df_i, how='left')
events['task+type'] = events['task'] + '/' + events['event_type']

# make annotations from events
onsets = events['latency_sec'] # in seconds! 
durations = events['duration_sec']
descriptions = events['task+type']

annot_all = mne.Annotations(onset=onsets, duration=durations, description=descriptions)

# %% select only reading events
# events_reading = events[events['task']=='reading']
# tmin = max([0, events_reading['latency_sec'].min()-60])
# tmax = min([EEG.times[-1],events_reading['latency_sec'].max()+60])
# EEG=EEG.crop(tmin=tmin, tmax=tmax)
# events_crop=events[events['latency_sec']>=EEG.times[0] & events['latency_sec']<=EEG.times[-1]]
# encode these events as annotation for epoching
EEG.set_annotations(annot_all)

# %% apply preprocessing to EEG
# reref to average
EEG.set_eeg_reference('average', projection=False)
# filter
EEG.filter(0.1, 40)
# resample to 100 Hz
EEG.resample(100)
# resample eveents: EEG sample needs to be recalculated. Use rounding as this is what events_from_annotations does
events['eeg_sample'] = (events['latency_sec']*EEG.info['sfreq']).round().astype(int)



#%% sanity check: basic rERP for FRP vs epoch
# event_id = 'Fixation_R'
# sel = events[events['event_type'].str.contains(event_id)][['event_type','task','description','latency_sec','duration_sec']]
# # convert to ANnotation
# sel_annot = mne.Annotations(onset=sel['latency_sec'], duration=sel['duration_sec'], description=sel['description'])
tmin, tmax = -0.3, 0.8
# convert to mne events (numpy array)
trl, trldict = mne.events_from_annotations(EEG, regexp='.*Fixation_R')
# drop duplicate rows - exact duplicates
e1 = len(trl)
trl = np.unique(trl, axis=0)
e2 = len(trl)
#  which rows have same onset time still
dupes = np.where(np.diff(trl[:,0])==0)
# drop duplicate rows - same start time (Event code might be different) 
trl = np.delete(trl, dupes, axis=0)
e3 = len(trl)

epochs = mne.Epochs(EEG, events=trl, event_id=trldict, tmin=-0.3, tmax=0.8, baseline=(-0.1, 0), preload=True, event_repeated='drop')
# rERP
rERPs = linear_regression_raw(EEG, events=trl, event_id=trldict, tmin=tmin, tmax=tmax)



# 
#%%  plot both results, and their difference
cond = 'reading/Fixation_R'

# plot ERP
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
params = dict(
    spatial_colors=True, show=False, ylim=dict(eeg=(-10, 10)), time_unit="s"
)
epochs[cond].average().plot(axes=ax1, **params)
rERPs[cond].plot(axes=ax2, **params)
contrast = mne.combine_evoked([rERPs[cond], epochs[cond].average()], weights=[1, -1])
contrast.plot(axes=ax3, **params)
ax1.set_title("Traditional averaging")
ax2.set_title("rERP")
ax3.set_title("Difference")
plt.show()

#%% RERP wioth covariate for surprisal and frequency and relative word position
# fixations only
fixations = events[events['event_type']=='Fixation_R']
# remove recalibartion fixations
fixations = fixations[~fixations['task'].str.contains('recal') | fixations['task'].str.contains('none')]
annot_fix = mne.Annotations(onset=fixations['latency_sec'], duration=fixations['duration_sec'], description=fixations['task+type'])
EEG.set_annotations(annot_fix)
trl_fix, trldict_fix = mne.events_from_annotations(EEG, regexp='.*Fixation_R')
# use latencies in trl to look up covariates in events
# drop  duplicates on eeg sample
dup_fixations = fixations[fixations.duplicated(subset='eeg_sample', keep=False)]['task+type']
# remove duplicated fixations, removing recal task version where possible
fixations = fixations.drop_duplicates(subset='eeg_sample', keep='first')
fixations = fixations.set_index('eeg_sample')

lexical_covariates = fixations.loc[trl_fix[:,0],[ 'surprisal', 'word_freq', 'relative_word_position']]
lexical_covariates = lexical_covariates.dropna(subset=['surprisal', 'word_freq', 'relative_word_position']).drop_duplicates()

# subselect trl and trldict to only contain fixations with valid covariates
trl_lex = trl_fix[np.isin(trl_fix[:,0], lexical_covariates.index),:]

# wrap solver in a function to make it a callable
ridge_alpha=1000
solver = lambda X, y: Ridge(solver='auto',alpha=ridge_alpha).fit(X, y).coef_ 

rERP_cov = linear_regression_raw(EEG, events=trl_lex, event_id=trldict, tmin=tmin, tmax=tmax, covariates=lexical_covariates, solver=solver)
rERP_2 = linear_regression_raw(EEG, events=trl_lex, event_id=trldict, tmin=tmin, tmax=tmax, solver=solver)
# plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

epochs[cond].average().plot(axes=ax1)
rERP_2[cond].plot(axes=ax2)
rERP_cov[cond].plot(axes=ax3)

# plot covariates
fig, ax = plt.subplots(3,1)
rERP_cov['surprisal'].plot(axes=ax[0])
rERP_cov['word_freq'].plot(axes=ax[1])
rERP_cov['relative_word_position'].plot(axes=ax[2])

#%% time-expanded model (fixation events only)
# create design matrix with 1 row per fixation epoch and 1 column perr value of cateorical predictor
fixations['eeg_index'] = np.searchsorted(EEG.times, fixations['latency_sec']) # might be off by one from orig eeg sample number... 
predictors = {}
predictors['intercept'] = np.ones(len(fixations))
onsets = fixations['eeg_index'].values  
# onsets = fixations[['latency_sec','eeg_index']]
# onsets['diff'] = onsets['eeg_index'] - onsets.index
for c in fixations['task'].unique():
    predictors[f'task:{c}'] = fixations['task']==c
lexical_covariates =[ 'surprisal', 'word_freq', 'relative_word_position']
for c in lexical_covariates:
    predictors[c] = fixations[c]
predictors = pd.DataFrame(predictors)
# drop redundant predictors so matrix is full rank
predictors.drop(columns=['task:none'], inplace=True)
X = predictors.to_numpy()

#%% time-expanded
# we want design matrix to have 1 element per timepoint in the continuous EEG
# so we need to expand the design matrix
# preallocate empty array and assign slice by chrisaycock https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

cols_timeExpanded = []
tmin, tmax = -0.3, 0.8  # Time window for expansion
sfreq = EEG.info['sfreq']
times = np.arange(tmin, tmax, 1/sfreq)
t_rel = np.arange(int(np.round(tmin*sfreq)), int(np.round(tmax*sfreq)))
cols_expanded = []
X_expanded_ = []
# intercept
cols_expanded.append('intercept')
X_expanded_.append(np.ones(len(EEG.times)))
for name in predictors.columns:
    if name == 'intercept':
        continue
    regressor = predictors[name]
    print(f'time-expanding regressor: {name}')
    regressor_expanded = np.zeros(len(EEG.times))
    regressor_expanded[onsets] = regressor
    # offset regressor for each epoch time point
    for ts, time in zip(t_rel, times):
        vec = shift(regressor_expanded, ts, fill_value=np.nan)
        name_exp = f'{name}_{time:.3f}'
        # append evc to numpy array
        X_expanded_.append(vec)
        cols_expanded.append(name_exp)

X_expanded = np.array(X_expanded_).T
# fill NaNs with 0
X_expanded = np.nan_to_num(X_expanded, nan=0)
# make it sparse
X_sparse = scipy.sparse.csr_matrix(X_expanded)
# plot the design matrix (downsample image for speed)
xte_plot = X_expanded[::10,:]
plt.figure()
plt.imshow(xte_plot, aspect='auto')
# Standardize the regressors
scaler = StandardScaler()
X_exp_scaled = scaler.fit_transform(X_expanded)

# %% fit model using linear_regrerssion on epoched data
# # convert to dummy single epoch
# dummy_EEG = mne.Epochs(EEG, mne.make_fixed_length_events(EEG, duration = EEG.times[-1], first_samp=False), tmin=0, tmax=EEG.times[-1], baseline=None, preload=True)
# beta_exp = mne.stats.linear_regression(dummy_EEG, X_expanded, names=cols_expanded)
epochs = mne.Epochs(EEG, events=trl_fix, event_id=trldict, tmin=tmin, tmax=tmax, baseline=(-0.1, 0), preload=True, event_repeated='drop')
X_exp_epoched = X_expanded[trl_fix[:,0],:]





#%% D'oh, linear_regression_raw does the timeexpanded stuff behind the scenes already
# fit with covariates, as for lexical rERP above, but retain non-IA fixation trials too & include saccade amplitude and poss other covaariates too
fixations = events[events['event_type']=='Fixation_R']
# remove recalibartion fixations
fixations = fixations[~fixations['task'].str.contains('recal') | fixations['task'].str.contains('none')]
# drop  duplicates on eeg sample
dup_fixations = fixations[fixations.duplicated(subset='eeg_sample', keep=False)]['task+type']
# remove duplicated fixations, removing recal task version where possible
fixations = fixations.drop_duplicates(subset='eeg_sample', keep='first')
fixations = fixations.set_index('eeg_sample')
annot_fix = mne.Annotations(onset=fixations['latency_sec'], duration=fixations['duration_sec'], description=fixations['task+type'])
EEG.set_annotations(annot_fix)
trl_fix, trldict_fix = mne.events_from_annotations(EEG, regexp='.*Fixation_R')
# use latencies in trl to look up covariates in events

# make imputations for missing values.
# FOr lexical variables use mean over STIMULUS set not ove rfixaitons/obs so it is the same for all ppts
fixations_unfilled = fixations.copy()
fixations['word_freq'] = fixations['word_freq'].fillna(ia_df['word_freq'].mean())
fixations['surprisal'] = fixations['surprisal'].fillna(ia_df['surprisal'].mean())
fixations['relative_word_position'] = fixations['relative_word_position'].fillna(ia_df['relative_word_position'].mean())
# # for these we essentially say that stop words nad punc are handles the same as non-word/ non-reading fixations, as it is only where these are False 
# # that interesting things are expected to happen w the lexical covariates
# fixations['stop_word'] = fixations['stop_word'].fillna(True).astype(bool) # 
# fixations['punctuation'] = fixations['punctuation'].fillna(True).astype(bool) # 
fixations['INBOUND_SAC_AMPLITUDE'] = fixations['INBOUND_SAC_AMPLITUDE'].fillna(fixations['INBOUND_SAC_AMPLITUDE'].mean()) # probably better to use some avg over whole datast? 
fixations['MW'] = fixations['MW'].fillna(1).astype(bool) # again when MW=0 is where interesting stuff happens, innattentuve reading is more likely to resemble nonreading? 

lexical_covariates = fixations.loc[trl_fix[:,0],[ 'surprisal', 'word_freq', 'relative_word_position']]
gaze_covariates = fixations.loc[trl_fix[:,0],[ 'INBOUND_SAC_AMPLITUDE']]#.fillna(0)
cognitive_covariates = fixations.loc[trl_fix[:,0],[ 'MW']]#.fillna(0)
covariates = pd.concat([lexical_covariates, gaze_covariates, cognitive_covariates], axis=1)
# wrap solver in a function to make it a callable
ridge_alpha=1000
solver = lambda X, y: Ridge(solver='auto',alpha=ridge_alpha).fit(X, y).coef_ 
# need to handle signle channe;: first dim should be of size 1 so add a null singleton dim to first dim
def ridge_solver(X, y): 
    res = Ridge(solver='auto',alpha=ridge_alpha).fit(X, y).coef_
    if len(res.shape)==1:
        res = np.expand_dims(res, axis=0)
    return res



# rERP_cov = linear_regression_raw(EEG, events=trl_fix, event_id=trldict, tmin=tmin, tmax=tmax, covariates=covariates, solver=solver)

# # viz fx for interaction between reading fixation and surprisal
# # plot
# fig, ax = plt.subplots(5, 1)
# rERP_cov['reading/Fixation_R'].plot(axes=ax[0])
# rERP_cov['surprisal'].plot(axes=ax[1])
# rERP_cov['word_freq'].plot(axes=ax[2])
# rERP_cov['relative_word_position'].plot(axes=ax[3])
# rERP_cov['MW'].plot(axes=ax[4])

#%% as above but 
# # 1. with separate condition for MW/no MW/not reading fixations
# 2. separate versions of lexical covariates for MW and no MW fixations
fixations['MW'] = fixations_unfilled['MW']
fixations['MW'] = fixations['MW'].fillna(-1).astype(int) # third condition for unknown MW label
fixations['task+type+MW'] = fixations['task+type'] + '/MW=' + fixations['MW'].astype(str)

annot_fixmw = mne.Annotations(onset=fixations['latency_sec'], duration=fixations['duration_sec'], description=fixations['task+type+MW'])
EEG.set_annotations(annot_fixmw)
trl_fixmw, trldict_fixmw = mne.events_from_annotations(EEG)

covariates.drop(columns=['MW'], inplace=True)
for c in lexical_covariates.columns:
    for mw in [-1,0,1]:
        covariates[f'{c}_MW={mw}'] = fixations.loc[trl_fixmw[:,0][fixations['MW']==mw], c]
        covariates[f'{c}_MW={mw}'] = covariates[f'{c}_MW={mw}'].fillna(ia_df[c].mean())
    covariates.drop(columns=c, inplace=True)

rERP_covmw = linear_regression_raw(EEG, 
    events=trl_fixmw, event_id=trldict_fixmw, 
    tmin=tmin, tmax=tmax, 
    reject={'eeg': 40e-5},
    # picks='CPz',
    covariates=covariates, solver=ridge_solver)

# plot contrast between MW and no MW for main effect 
fig, ax = plt.subplots(3, 1)
rERP_covmw['reading/Fixation_R/MW=0'].plot(axes=ax[0], picks='CPz')
rERP_covmw['reading/Fixation_R/MW=1'].plot(axes=ax[1], picks='CPz')
contrast = mne.combine_evoked([rERP_covmw['reading/Fixation_R/MW=0'], rERP_covmw['reading/Fixation_R/MW=1']], weights=[1, -1])
contrast.plot(axes=ax[2], picks='CPz')

# plot surprisal effect for MW and no MW separately
fig, ax = plt.subplots(2, 1)
rERP_covmw['surprisal_MW=0'].plot(axes=ax[0], picks='CPz')
rERP_covmw['surprisal_MW=1'].plot(axes=ax[1], picks='CPz')

# same plot different conditions
fig, ax = plt.subplots(2, 1)
condlist = ['reading/Fixation_R/MW=-1','reading/Fixation_R/MW=0', 'reading/Fixation_R/MW=1']
mne.viz.plot_compare_evokeds([rERP_covmw[c] for c in condlist], picks='CPz', axes=ax[0])
condlist = ['surprisal_MW=-1','surprisal_MW=0', 'surprisal_MW=1']
mne.viz.plot_compare_evokeds([rERP_covmw[c] for c in condlist], picks='CPz', axes=ax[1])
#%% save regression erps
rERP_covmw.save(os.path.join(dir_out, f'{pID}_rERP_covmw-epo.fif'))
# append to list of all subjects
rERP_covmw_ALL.append(rERP_covmw)