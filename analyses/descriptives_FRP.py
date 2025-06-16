# %%
import pandas as pd
import numpy as np
import re
import os
import mne
# look in overall resutls folder and compare skip_reasons.csv between analysis verisons

dir_out = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EML Rosy/Data/EEG_processed/FRP_TRF_lexical_v12_alpha1_decim1'

# get pIDs used 
pIDs  = os.listdir(dir_out)
pIDs = [p for p in pIDs if re.match(r'EML1_\d{3}_rERP-evk', p)]
# get jsut the pID part
pIDs = [re.search(r'EML1_\d{3}', p).group(0) for p in pIDs]
# sort
pIDs.sort()

# %% load behavioral data

beh_df = pd.read_csv(
    '~/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/Processed/Behaviour/EML1_page_level.csv'
)  # comp and MW scores
# get MW rate per ppt
# select only ParticipantID in pIDs
beh_df = beh_df[beh_df['ParticipantID'].isin(pIDs)]
mw = beh_df.groupby('ParticipantID')['MW'].mean().reset_index()
# which participant outside rance 0.1-0.9 MW
mw = mw[(mw['MW'] >= 0.1) & (mw['MW'] <= 0.9)]
# participants to keep
pIDs = mw['ParticipantID'].tolist()

# %% load stats 

rERPs = {}
stats = {'pID':[],'reading_fixation': [], 'surprisal': []}
for pID in pIDs:
    rERP = mne.read_evokeds(
        os.path.join(dir_out, f'{pID}_rERP-evk.fif'), verbose='ERROR'
    )
    rERP = {evk.comment: evk for evk in rERP}

    rERPs[pID] = rERP
    nfix = rERP['reading/Fixation'].nave
    ncontent = rERP['surprisal'].nave
    print(f'{pID}: nfix={nfix}, ncontent={ncontent}')
    stats['reading_fixation'].append(nfix)
    stats['surprisal'].append(ncontent)
    stats['pID'].append(pID)



pID_stats = pd.DataFrame({
    'ParticipantID': stats['pID'],
    'n_fixations': stats['reading_fixation'],
    'n_content_words': stats['surprisal'],
})
# merge in MW on ParticipantID and pID
pID_stats = pID_stats.merge(mw, on='ParticipantID', how='left')
pID_stats.to_csv(
    os.path.join(dir_out, 'pID_level_stats.csv'), index=False
)
# %% get mean and sd over pIDs (just for numericcal cols   )
# Select only numerical columns for mean and std
numeric_cols = pID_stats.select_dtypes(include=[np.number]).columns
mean_stats = pID_stats[numeric_cols].mean()
sd_stats = pID_stats[numeric_cols].std()
# combine into one table
stats_table = pd.DataFrame({
    'mean': mean_stats,
    'sd': sd_stats
}).T
# export stats table
stats_table.to_csv(
    os.path.join(dir_out, 'pID_stats_summary.csv'), index=False
)

# %% histograms for MW and n_fixations, n_content_words
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
pID_stats['MW'] = pID_stats['MW'].astype(float)  # ensure MW is float
sns.histplot(pID_stats['MW'], bins=bins,  alpha=0.7)
plt.xlabel('MW Rate')

plt.subplot(1, 3, 2)
sns.histplot(pID_stats['n_fixations'], bins=20,  alpha=0.7)
plt.xlabel('Number of Fixations')

plt.subplot(1, 3, 3)
sns.histplot(pID_stats['n_content_words'], bins=20,  alpha=0.7)
plt.xlabel('Fixations on Content Words')

plt.tight_layout()
plt.savefig(os.path.join(dir_out, 'pID_stats_histograms.png'))
plt.show()

# %%
