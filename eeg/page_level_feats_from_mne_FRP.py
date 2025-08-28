#%%
import mne
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def get_stats_for_win(win, data, times, rms=False):
    # data is trials * time
    sel = np.where((win[0] <= np.array(times)) & (np.array(times) <= win[1]))[0]
    y = data[:,sel]
    win_stats={}
    win_stats['max'] = np.max(y, axis=1)
    win_stats['mean'] = np.mean(y, axis=1)
    ix = np.argmax(y, axis=1)
    win_stats['max_lat'] = [times[sel[i]] for i in ix]
    if not rms: # only positive feats make sense for RMS
        win_stats['min'] = np.min(y, axis=1)
        win_stats['max_abs'] = np.max(np.abs(y), axis=1)
        ix = np.argmax(np.abs(y), axis=1)
        win_stats['max_abs_lat'] = [times[sel[i]] for i in ix]
        win_stats['min_lat'] = [times[sel[i]] for i in np.argmin(y, axis=1)]
        win_stats['zero_crossings'] = np.sum(np.diff(np.sign(y), axis=1) != 0, axis=1)
    return win_stats
def top_3_cols_with_nans(df):
    # find top 3 columns with most NaNs and print both the name of the col and the nan count
    nans = df.isna().sum()
    nans = nans.sort_values(ascending=False)
    for i in range(3):
        print(f'{nans.index[i]}: {nans[i]} NaNs')
    return nans

# %% Get FRP data
dir_in = '/Volumes/Blue1TB/EEG_processed/FRP_GLM_allfix_v2_decim10'
pIDs =  [re.findall(r'EML1_\d{3}', f)[0] for f in os.listdir(dir_in) if 'dcFRP_epochs_allchannels' in f]
pIDs = sorted(list(set(pIDs)))
featdir = 'FRP_stats'
os.makedirs(os.path.join(dir_in,featdir), exist_ok=True)
ia_label_mapping = pd.read_csv('../info/ia_label_mapping_opt_surprisal.csv')
min_word_freq = np.min(ia_label_mapping.loc[ia_label_mapping['word_freq'] > 0, 'word_freq'])
# log_word_freq_fill_value = np.log(min_word_freq)-1
log_word_freq_fill_value = None # actually I htink it makes more sense to fill non words with mean later - nonword IAs aren't like super rare words.
FRPall = []
FRPavg = []
channels = ['CPz', 'FCz', 'AFF5h', 'AFF6h', 'CCP5h', 'CCP6h', 'PPO9h', 'PPO10h']
channel_combos = {'AFFave':['AFF5h','AFF6h'],'CCPave':['CCP5h','CCP6h'],'PPOave':['PPO9h','PPO10h'],
'RMS_all':['CPz','FCz','AFF5h','AFF6h','CCP5h','CCP6h','PPO9h','PPO10h']} # string 'rms' in combo key will trigger RMS instead ofmean as agg func

windows = {'P1': [70, 120], 'N1': [140, 280], 'N400': [300, 500]}
components = {'P1': {'latency': [70, 120],'channels':['PPO9h', 'PPO10h']}, 
            #    'N1':{'latency': [180, 220], 'channels':['CCP5h','PPO9h']},
              'P2': {'latency':[140, 220],'channels':['CPz', 'FCz']},
              'N400': {'latency':[250, 500],'channels':['CPz']},        # 300-500 is typical, Boudewyn use 300-600, Frank Aumestiere use 250-450
            #   'P600': {'latency':[500, 800],'channels':['CPz', 'FCz']}, 
}
#%%
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
            nans=top_3_cols_with_nans(data)
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
        if 'rms' in combo_label.lower():
            data = np.sqrt(np.mean(data_array**2, axis=0))
        else:   
            data = np.mean(data_array, axis=0)
        for win_label, win in windows.items():
            res = get_stats_for_win(win, data, times, rms=('rms' in combo_label.lower()))
            # add columns to stats, prepend channel name an windown name
            for k, v in res.items():
                cols_to_add[f'{win_label}_{combo_label}_{k}'] = v
    # same for specific components
    for comp_label, comp in components.items():
        win = comp['latency']
        data_array = []
        for channel in comp['channels']:
            data = data_dict[channel].values
            data_array.append(data)
        data_array = np.array(data_array)
        data = np.mean(data_array, axis=0)
        res = get_stats_for_win(win, data, times, rms=False)
            # add columns to stats, prepend component name and window name
        for k, v in res.items():
            cols_to_add[f'{comp_label}_{k}'] = v

    # z score the EEG min, mean and max features per participant
    to_scale = [col for col in cols_to_add.keys() if re.search(r'mean|min|max', col) and not re.search(r'log|lat', col)]
    for col in to_scale:
        cols_to_add[col+'_Z'] = (cols_to_add[col] - cols_to_add[col].mean()) / cols_to_add[col].std()

    stats = pd.concat([stats, pd.DataFrame(cols_to_add)], axis=1)
    data_array =np.array( [data_dict[channel].values for channel in channels])
    FRPavg = np.mean(data_array, axis=1) # TODO: check axis
    FRPall.append(FRPavg)
    stats['ParticipantID'] = pID
    stats = pd.DataFrame(stats)
    # log transform surprisal and freq
    stats['log_word_freq'] = np.where(stats['word_freq'] > 0, np.log(stats['word_freq']), log_word_freq_fill_value)
    
    # # count na rows and drop them
    # n_na = np.sum(stats.isna().any(axis=1))
    # if n_na > 0:    
    #     print(f'{pID} has {n_na} rows with NaNs')
    #     stats = stats.dropna()

    stats.to_csv(os.path.join(dir_in,featdir, f'{pID}_dcFRP_stats.csv'), index=False)
    stats_all.append(stats)

stats_all = pd.concat(stats_all)
stats_all.to_csv(os.path.join(dir_in,featdir, f'ALL_dcFRP_stats.csv'), index=False)

# %% make df with layout for Megan
# read in stats per subj
stats_all = []
for pID in pIDs:
    stats = pd.read_csv(os.path.join(dir_in,featdir, f'{pID}_dcFRP_stats.csv'))
    stats_all.append(stats)
stats_all = pd.concat(stats_all)

stats_fmt = stats_all.rename(columns={'identifier': 'EVENT','task': 'TrialType'})

# split event on letter/digit boundary to get text and pagenum cols
stats_fmt['Text'] = [re.findall(r'[a-zA-Z]+', e)[0] for e in stats_fmt['EVENT']]
stats_fmt['PageNum'] = [re.findall(r'\d+', e)[0] for e in stats_fmt['EVENT']]
identifiers = ['ParticipantID','Text','PageNum','TrialType','EVENT']
stats_fmt['logFixDur'] = np.log(stats_fmt['duration_sec'])
# select key features
feats = ['N400_CPz_mean', 'N400_CPz_min', 'N400_CPz_min_lat', 'N400_CPz_max_abs','N400_CPz_max_abs_lat',
        'N400_FCz_mean', 'N400_FCz_min', 'N400_FCz_min_lat', 'N400_FCz_max_abs','N400_FCz_max_abs_lat',
        'P1_PPOave_mean', 'P1_PPOave_max', 'P1_PPOave_max_lat', 'P1_PPOave_max_abs', 'P1_PPOave_max_abs_lat',
        'N1_PPOave_mean', 'N1_PPOave_min', 'N1_PPOave_min_lat', 'N1_PPOave_max_abs', 'N1_PPOave_max_abs_lat',
        'N400_RMS_all_mean', 'N400_RMS_all_max', 'N400_RMS_all_max_lat',
        'N1_RMS_all_mean',  'N1_RMS_all_max', 'N1_RMS_all_max_lat',
        'P1_RMS_all_mean', 'P1_RMS_all_max' , 'P1_RMS_all_max_lat',
        'duration_sec','fix_pupilAvg', 'logFixDur',
        'surprisal', 'log_word_freq', 'punctuation','stop_word','fix_pageIndex'
]
stats_fmt = stats_fmt[identifiers + feats]
stats_fmt.to_csv(os.path.join(dir_in,featdir, f'ALL_dcFRP_stats_formatted.csv'), index=False)
# compute page-level averages of these feats
stats_pglevel = stats_fmt.groupby(identifiers).mean().reset_index()

# compute correlations per page for certain feature combos
correlation_feats = { 
    'surprisal~N400_CPz_max_abs': ['N400_CPz_max_abs', 'surprisal'],
    'log_word_freq~N400_CPz_max_abs': ['N400_CPz_max_abs', 'log_word_freq'],
    'surprisal~N400_CPz_mean': ['N400_CPz_mean', 'surprisal'],
    'log_word_freq~N400_CPz_mean': ['N400_CPz_mean', 'log_word_freq'],
    'surprisal~N400_CPz_min': ['N400_CPz_min', 'surprisal'],
    'log_word_freq~N400_CPz_min': ['N400_CPz_min', 'log_word_freq'],
    'surprisal~N400_CPz_min_lat': ['N400_CPz_min_lat', 'surprisal'],
    'log_word_freq~N400_CPz_min_lat': ['N400_CPz_min_lat', 'log_word_freq'],
    'surprisal~N400_FCz_max_abs': ['N400_FCz_max_abs', 'surprisal'],
    'log_word_freq~N400_FCz_max_abs': ['N400_FCz_max_abs', 'log_word_freq'],
    'surprisal~N400_FCz_mean': ['N400_FCz_mean', 'surprisal'],
    'log_word_freq~N400_FCz_mean': ['N400_FCz_mean', 'log_word_freq'],
    'surprisal~N400_FCz_min': ['N400_FCz_min', 'surprisal'],
    'log_word_freq~N400_FCz_min': ['N400_FCz_min', 'log_word_freq'],
    'surprisal~N400_FCz_min_lat': ['N400_FCz_min_lat', 'surprisal'],
    'log_word_freq~N400_FCz_min_lat': ['N400_FCz_min_lat', 'log_word_freq'],
    'surprisal~P1_PPOave_max_abs': ['P1_PPOave_max_abs', 'surprisal'],
    'log_word_freq~P1_PPOave_max_abs': ['P1_PPOave_max_abs', 'log_word_freq'],
    'surprisal~P1_PPOave_mean': ['P1_PPOave_mean', 'surprisal'],
    'log_word_freq~P1_PPOave_mean': ['P1_PPOave_mean', 'log_word_freq'],
    'surprisal~P1_PPOave_max': ['P1_PPOave_max', 'surprisal'],
    'log_word_freq~P1_PPOave_max': ['P1_PPOave_max', 'log_word_freq'],
    'surprisal~P1_PPOave_max_lat': ['P1_PPOave_max_lat', 'surprisal'],
    'log_word_freq~P1_PPOave_max_lat': ['P1_PPOave_max_lat', 'log_word_freq'],
    'surprisal~N1_PPOave_max_abs': ['N1_PPOave_max_abs', 'surprisal'],
    'log_word_freq~N1_PPOave_max_abs': ['N1_PPOave_max_abs', 'log_word_freq'],
    'surprisal~N1_PPOave_mean': ['N1_PPOave_mean', 'surprisal'],
    'log_word_freq~N1_PPOave_mean': ['N1_PPOave_mean', 'log_word_freq'],
    'surprisal~N1_PPOave_min': ['N1_PPOave_min', 'surprisal'],
    'log_word_freq~N1_PPOave_min': ['N1_PPOave_min', 'log_word_freq'],
    'surprisal~N1_PPOave_min_lat': ['N1_PPOave_min_lat', 'surprisal'],
    'log_word_freq~N1_PPOave_min_lat': ['N1_PPOave_min_lat', 'log_word_freq'],
    'surprisal~duration_sec': ['duration_sec', 'surprisal'],
    'log_word_freq~duration_sec': ['duration_sec', 'log_word_freq'],
    'surprisal~fix_pupilAvg': ['fix_pupilAvg', 'surprisal'],
    'log_word_freq~fix_pupilAvg': ['fix_pupilAvg', 'log_word_freq'],
    'P1_PPOave_mean~logFixDur': ['P1_PPOave_mean', 'logFixDur'],
    'N1_PPOave_mean~logFixDur': ['N1_PPOave_mean', 'logFixDur'],
    'N400_CPz_mean~logFixDur': ['N400_CPz_mean', 'logFixDur'],
    'P1_PPOave_max~logFixDur': ['P1_PPOave_max', 'logFixDur'],
    'N1_PPOave_min~logFixDur': ['N1_PPOave_min', 'logFixDur'],
    'N400_CPz_min~logFixDur': ['N400_CPz_min', 'logFixDur'],
    'P1_PPOave_max_lat~logFixDur': ['P1_PPOave_max_lat', 'logFixDur'],
    'N1_PPOave_min_lat~logFixDur': ['N1_PPOave_min_lat', 'logFixDur'],
    'N400_CPz_min_lat~logFixDur': ['N400_CPz_min_lat', 'logFixDur'],

}
for k, v in correlation_feats.items():
    # group by identifiers and compute correlation
    stats_pglevel[k] = stats_fmt.groupby(identifiers)[v].corr().iloc[0::2,-1].values

stats_pglevel.to_csv(os.path.join(dir_in,featdir, f'ALL_dcFRP_stats_page_level.csv'), index=False)

# %% plot some key pairs of features in scatter plots
fig, axs = plt.subplots(2, 3, figsize=(12,12))
to_plot=[['fix_pupilAvg', 'surprisal'],
['fix_pupilAvg', 'log_word_freq'],
['duration_sec', 'surprisal'],
['duration_sec', 'log_word_freq'],
['N400_CPz_mean_Z', 'surprisal'],
['N400_CPz_mean_Z', 'log_word_freq'],
]
for i, (y,x) in enumerate(to_plot):
    ax = axs.flatten()[i]
    for pID in pIDs[0:10]:
        stats = stats_all[stats_all['ParticipantID'] == pID]
        ax.scatter(stats[x], stats[y], alpha=0.1, label=pID)    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
# %% plot some key pairs of EEG features in scatter plots
fig, axs = plt.subplots(2, 3, figsize=(12,12))
to_plot=[['N400_CPz_max_abs_Z', 'surprisal'],
['N400_CPz_max_abs_Z', 'log_word_freq'],
['N400_CPz_max_abs_lat', 'surprisal'],
['N400_CPz_max_abs_lat', 'log_word_freq'],
['N400_CPz_mean_Z', 'surprisal'],
['N400_CPz_mean_Z', 'log_word_freq'],
]
for i, (y,x) in enumerate(to_plot):
    ax = axs.flatten()[i]
    for pID in pIDs[0:10]:
        statsp = stats_all[stats_all['ParticipantID'] == pID]
        ax.scatter(statsp[x], statsp[y], alpha=0.1, label=pID)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

# save plot
fig.savefig(os.path.join(dir_in,featdir, f'key_feat_scatters.png'))
# %% same as above but color code by participant
fig, axs = plt.subplots(2, 3, figsize=(12,12))
to_plot=[['N400_CPz_max_abs', 'surprisal'],
['N400_CPz_max_abs', 'log_word_freq'],
['N400_CPz_max_abs_lat', 'surprisal'],
['N400_CPz_max_abs_lat', 'log_word_freq'],
['N400_CPz_mean', 'surprisal'],
['N400_CPz_mean', 'log_word_freq'],
]
for i, (y,x) in enumerate(to_plot):
    ax = axs.flatten()[i]
    for pID in pIDs[0:10]:
        statsp = stats_all[stats_all['ParticipantID'] == pID]
        ax.scatter(statsp[x], statsp[y], alpha=0.1, label=pID)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
fig.savefig(os.path.join(dir_in,featdir, f'key_feat_scatters_byPID.png'))
# %%
