#%%
import mne
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def get_stats_for_win(win, data, times):
    # data is trials * time
    sel = np.where((win[0] <= np.array(times)) & (np.array(times) <= win[1]))[0]
    y = data[:,sel]
    win_stats={}
    win_stats['max'] = np.max(y, axis=1)
    win_stats['min'] = np.min(y, axis=1)
    win_stats['max_abs'] = np.max(np.abs(y), axis=1)
    ix = np.argmax(y, axis=1)
    win_stats['max_lat'] = [times[sel[i]] for i in ix]
    ix = np.argmax(np.abs(y), axis=1)
    win_stats['max_abs_lat'] = [times[sel[i]] for i in ix]
    win_stats['min_lat'] = [times[sel[i]] for i in np.argmin(y, axis=1)]
    win_stats['zero_crossings'] = np.sum(np.diff(np.sign(y), axis=1) != 0, axis=1)
    win_stats['mean'] = np.mean(y, axis=1)
    return win_stats

# %% Get FRP data
dir_in = '/Volumes/Blue1TB/EEG_processed/FRP_GLM_simple_decim10'
pIDs =  [re.findall(r'EML1_\d{3}', f)[0] for f in os.listdir(dir_in) if 'dcFRP_epochs_allchannels' in f]
pIDs = sorted(list(set(pIDs)))
featdir = 'FRP_stats'
os.makedirs(os.path.join(dir_in,featdir), exist_ok=True)
FRPall = []
FRPavg = []
channels = ['CPz', 'FCz', 'AFF5h', 'AFF6h', 'CCP5h', 'CCP6h', 'PPO9h', 'PPO10h']
channel_combos = {'AFFave':['AFF5h','AFF6h'],'CCPave':['CCP5h','CCP6h'],'PPOave':['PPO9h','PPO10h']}
windows = {'P1': [70, 120], 'N1': [140, 280], 'N400': [300, 500]}

stats_all = []
for pID in pIDs:
    df = pd.read_csv(os.path.join(dir_in, f'{pID}_dcFRP_epochs_allchannels.csv')).dropna(subset=['identifier']).drop(columns=['Unnamed: 0'])
    time_cols = [col for col in df.columns if re.search(r'\d{1,3}ms', col)]
    stats = df.copy().drop(columns = time_cols)
    data_dict = {}
    cols_to_add = {}
    for c, channel in enumerate(channels):
        # select 
        # remove individual timepoint columns
        # subset by channel
        data_cols = [col for col in time_cols if channel in col]
        data = df[data_cols]
        data_dict[channel] = data # for later use w combos
        # check for NaNs
        if np.any(np.isnan(data)):
            print(f'{pID} {channel} has {np.sum(np.isnan(data))} NaNs')
            #where are the nans
        times = [int(re.findall(r'(-?\d{1,3})ms', col)[0]) for col in data_cols]
        for win_label, win in windows.items():
            res = get_stats_for_win(win, data.values, times)
            # add columns to stats, prepend channel name an windown name
            for k, v in res.items():
                cols_to_add[f'{win_label}_{channel}_{k}'] = v
    # same for combo channels
    for (combo_label, channs) in channel_combos.items():
        data_array = []
        for channel in channs:
            data = data_dict[channel].values
            data_array.append(data)
        data_array = np.array(data_array)
        data = np.mean(data_array, axis=0)
        for win_label, win in windows.items():
            res = get_stats_for_win(win, data, times)
            # add columns to stats, prepend channel name an windown name
            for k, v in res.items():
                cols_to_add[f'{win_label}_{combo_label}_{k}'] = v
    stats = pd.concat([stats, pd.DataFrame(cols_to_add)], axis=1)
    data_array =np.array( [data_dict[channel].values for channel in channels])
    FRPavg = np.mean(data_array, axis=1) # TODO: check axis
    FRPall.append(FRPavg)
    stats['ParticipantID'] = pID
    stats = pd.DataFrame(stats)
    stats.to_csv(os.path.join(dir_in,featdir, f'{pID}_dcFRP_stats.csv'), index=False)
    stats_all.append(stats)
stats_all = pd.concat(stats_all)
# stats_all.to_csv(os.path.join(dir_in,featdir, f'ALL_dcFRP_stats.csv'), index=False)
# %% make df with layout for Megan
# read in stats per subj
stats_all = []
for pID in pIDs:
    stats = pd.read_csv(os.path.join(dir_in,featdir, f'{pID}_dcFRP_stats.csv'))
    stats_all.append(stats)
stats_all = pd.concat(stats_all)

stats_fmt = stats_all.rename(columns={'identifier': 'EVENT','task': 'TrialType'})
stats_fmt = stats_fmt[stats_fmt['TrialType'] == 'reading']
stats_fmt = stats_fmt[stats_fmt['stop_word'] == 0]
# split event on letter/digit boundary to get text and pagenum cols
stats_fmt['Text'] = [re.findall(r'[a-zA-Z]+', e)[0] for e in stats_fmt['EVENT']]
stats_fmt['PageNum'] = [re.findall(r'\d+', e)[0] for e in stats_fmt['EVENT']]
identifiers = ['ParticipantID','Text','PageNum','TrialType','EVENT']

# select key features
feats = ['N400_CPz_mean', 'N400_CPz_min', 'N400_CPz_min_lat', 'N400_CPz_max_abs','N400_CPz_max_abs_lat',
        'N400_FCz_mean', 'N400_FCz_min', 'N400_FCz_min_lat', 'N400_FCz_max_abs','N400_FCz_max_abs_lat',
        'P1_PPOave_mean', 'P1_PPOave_max', 'P1_PPOave_max_lat', 'P1_PPOave_max_abs', 'P1_PPOave_max_abs_lat',
        'N1_PPOave_mean', 'N1_PPOave_max', 'N1_PPOave_max_lat', 'N1_PPOave_max_abs', 'N1_PPOave_max_abs_lat',

        'surprisal', 'word_freq'
]
stats_fmt = stats_fmt[identifiers + feats]

# compute page-level averages of these feats
stats_pglevel = stats_fmt.groupby(identifiers).mean().reset_index()

# compute correlations per page for certain feature combos
correlation_feats = { 
    'surprisal~N400_CPz_max_abs': ['N400_CPz_max_abs', 'surprisal'],
    'word_freq~N400_CPz_max_abs': ['N400_CPz_max_abs', 'word_freq'],
    'surprisal~N400_CPz_mean': ['N400_CPz_mean', 'surprisal'],
    'word_freq~N400_CPz_mean': ['N400_CPz_mean', 'word_freq'],
    'surprisal~N400_CPz_min': ['N400_CPz_min', 'surprisal'],
    'word_freq~N400_CPz_min': ['N400_CPz_min', 'word_freq'],
    'surprisal~N400_CPz_min_lat': ['N400_CPz_min_lat', 'surprisal'],
    'word_freq~N400_CPz_min_lat': ['N400_CPz_min_lat', 'word_freq'],
    'surprisal~N400_FCz_max_abs': ['N400_FCz_max_abs', 'surprisal'],
    'word_freq~N400_FCz_max_abs': ['N400_FCz_max_abs', 'word_freq'],
    'surprisal~N400_FCz_mean': ['N400_FCz_mean', 'surprisal'],
    'word_freq~N400_FCz_mean': ['N400_FCz_mean', 'word_freq'],
    'surprisal~N400_FCz_min': ['N400_FCz_min', 'surprisal'],
    'word_freq~N400_FCz_min': ['N400_FCz_min', 'word_freq'],
    'surprisal~N400_FCz_min_lat': ['N400_FCz_min_lat', 'surprisal'],
    'word_freq~N400_FCz_min_lat': ['N400_FCz_min_lat', 'word_freq'],
    'surprisal~P1_PPOave_max_abs': ['P1_PPOave_max_abs', 'surprisal'],
    'word_freq~P1_PPOave_max_abs': ['P1_PPOave_max_abs', 'word_freq'],
    'surprisal~P1_PPOave_mean': ['P1_PPOave_mean', 'surprisal'],
    'word_freq~P1_PPOave_mean': ['P1_PPOave_mean', 'word_freq'],
    'surprisal~P1_PPOave_max': ['P1_PPOave_max', 'surprisal'],
    'word_freq~P1_PPOave_max': ['P1_PPOave_max', 'word_freq'],
    'surprisal~P1_PPOave_max_lat': ['P1_PPOave_max_lat', 'surprisal'],
    'word_freq~P1_PPOave_max_lat': ['P1_PPOave_max_lat', 'word_freq'],
    'surprisal~N1_PPOave_max_abs': ['N1_PPOave_max_abs', 'surprisal'],
    'word_freq~N1_PPOave_max_abs': ['N1_PPOave_max_abs', 'word_freq'],
    'surprisal~N1_PPOave_mean': ['N1_PPOave_mean', 'surprisal'],
    'word_freq~N1_PPOave_mean': ['N1_PPOave_mean', 'word_freq'],
    'surprisal~N1_PPOave_max': ['N1_PPOave_max', 'surprisal'],
    'word_freq~N1_PPOave_max': ['N1_PPOave_max', 'word_freq'],
    'surprisal~N1_PPOave_max_lat': ['N1_PPOave_max_lat', 'surprisal'],
    'word_freq~N1_PPOave_max_lat': ['N1_PPOave_max_lat', 'word_freq'],
}
for k, v in correlation_feats.items():
    # group by identifiers and compute correlation
    stats_pglevel[k] = stats_fmt.groupby(identifiers)[v].corr().iloc[0::2,-1].values

stats_pglevel.to_csv(os.path.join(dir_in,featdir, f'ALL_dcFRP_stats_page_level.csv'), index=False)

# %%
