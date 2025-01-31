# %%
from scipy import io
import h5py
import mat73
from mne.io import read_raw_eeglab
import os
import torch
import pandas as pd
import numpy as np

def load_mat_file(file_name):
    try:
        data = io.loadmat(file_name, squeeze_me=True, struct_as_record=False)
        print(f'Loaded data using io.loadmat')
    except Exception as e:
        print(f'Failed to load using io.loadmat: {e}')
        try:
            data = mat73.loadmat(file_name)
            print(f'Loaded data using mat73.loadmat')
        except:
            print(f'Failed to load using mat73.loadmat')
            data = h5py.File(file_name, 'r')
    print(f'loaded data of type {type(data)}, length {len(data)}')
    if isinstance(data, dict):
        print(f'keys: {data.keys()}')
    return data

# %% load some example EEG data
path_to_zuco = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EEG-Gaze/ZuCo/osfstorage'
file_name = os.path.join(path_to_zuco,"task1- SR/Matlab files/resultsZAB_SR.mat")
# data = io.loadmat(file_name, squeeze_me=True, struct_as_record=False)
data = load_mat_file(file_name)['sentenceData']

# %% browse fields
print(len(data)) # n sentences
sent_ix = 0
sent = data[sent_ix]
print(sent._fieldnames)
# raw EEG
sent_EEG = sent.rawData
sent_EEG.shape # dim: n channels x n timepoints
print(sent.content)
print(sent.word[0].content)
print(sent.word[0]._fieldnames)
print(sent.word[0].fixPositions) # indices of fixations for this word
print(sent.word[0].nFixations) # indices of fixations for this word
print(sent.word[0].rawET[0]) # n fix. * timepoints? Some are 1d, others have 2d with first dim=4
print(sent.word[0].rawEEG[0].shape) # n fix. * timepoints? Some are 1d, others have 2d with first dim=4

# %% get fixation sequence from sentence
fix_seq = []
for i, word in enumerate(sent.word):
    print(word.content)
    if word.nFixations > 0:
        if word.nFixations == 1:
            fix_seq.extend((i,word.fixPositions))
        else:
            for j in range(word.nFixations):
                fix_seq.append((i,word.fixPositions[j]))
# now get a list of word indices ordered by fixation indices (note this doesnt include fixations outside of word)
# drop entreis without a fixation
fix_seq = [f for f in fix_seq if isinstance(f, tuple)]
fix_seq = sorted(fix_seq, key=lambda x: x[1])
fixations = pd.DataFrame(fix_seq, columns=['word_ix','fix_ix'])
fixations.set_index('fix_ix', inplace=True)
fixations['count_on_word']= fixations.groupby('word_ix').cumcount()
# look up fixation durations
fix_EEG = []
for i, row in fixations.iterrows():
    fix_EEG.append(sent.word[row['word_ix']].rawEEG[row['count_on_word']])
word_fixation_sequence = {'content': sent.content, 'fixations': fixations, 'EEG': fix_EEG}

# %%


# # Load an EEG file to get channels labels as .mat files only have indices
# eegfile = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EEG-Gaze/ZuCo/osfstorage/task3 - TSR/Preprocessed/ZAB/gip_ZAB_TSR1_EEG.mat'
# eeg = load_mat_file(eegfile)
# channels = eeg['EEG']['chanlocs']
# channel_names = [c['labels'] for c in eeg['EEG']['chanlocs']]
# # save to txt file
# with open('ZuCo_channels.txt', 'w') as f:
#     for i,item in enumerate(channel_names):
#         f.write(f"{i},{item}\n")

# egi1010 = pd.read_csv('EGI-1010.txt', sep='\s')
# tmp = pd.concat([egi1010[["10-10","HydroCel","ArcLength"]],
# egi1010[["10-10.1","HydroCel.1","ArcLength.1"]].rename(
#     columns={"10-10.1":"10-10","HydroCel.1":"HydroCel", "ArcLength.1":"ArcLength"})],axis=0, ignore_index=True).dropna()  
# tmp['HydroCel'] ='E' + tmp['HydroCel'].astype(int).astype(str)
# tmp.to_csv('EGI-1010.csv', index=False)
zuco_channels = pd.read_csv('ZuCo_channels.txt', names=['zuco_chan_ix','HydroCel'])
egi_1010 = pd.read_csv('EGI-1010.csv')

# %% prepare ZuCo for BENDR: channel selection
channels_1020 = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2']
# get egi channels which overlap with these
channels_1020 = pd.DataFrame(channels_1020, columns=['10-10'])
channels_1020 = pd.merge(channels_1020, egi_1010, on='10-10', how='left')
channels_1020 = pd.merge(channels_1020, zuco_channels, on='HydroCel', how='left')
# look up indices of these in the ZuCo data
use_chan_ix = channels_1020['zuco_chan_ix'].dropna().astype(int).values


# %% bendr encoder
from get_bendr_feats import init_pretrained_bendr_encoder
bendr_encoder, bendr_contextualizer = init_pretrained_bendr_encoder()

# EEG for an example fixation
egeeg = sent.word[0].rawEEG[0]
egeeg = egeeg[:,use_chan_ix]
# TODO: resample: zuco uses 500 , BENDR uses 256

# add additional channel for scale, set to all 0 for now
egeeg = np.concatenate([egeeg, np.zeros((egeeg.shape[0],1))], axis=1).T

egeeg_batch = torch.tensor(egeeg).unsqueeze(0).float()


with torch.no_grad():
    enc = bendr_encoder(torch.tensor(egeeg_batch))

# encode all EEGs in word_fixation_sequence
word_fixation_sequence['bendr_EEG_enc'] = []
eeg_batch = []
max_len = 0
for eeg in word_fixation_sequence['EEG']:
    # print(eeg.shape)
    eeg = eeg[use_chan_ix,:]
    eeg = np.concatenate([eeg, np.zeros((1,eeg.shape[1]))], axis=0)
    print(eeg.shape)
    eeg_batch.append(eeg)
    max_len = eeg.shape[1] if eeg.shape[1] > max_len else max_len
# pad to max length of eegs
eeg_batch = [torch.nn.functional.pad(torch.tensor(eeg), (0,max_len-eeg.shape[1])) for eeg in eeg_batch]
# make batch of dims n_fixations x n_channels x n_timepoints
eeg_batch = torch.stack(eeg_batch).float()
with torch.no_grad():
    enc_batch = bendr_encoder(eeg_batch)
# unpack the encs into a list of fixaations
enc_batch_unpacked = [enc_batch[i,:].numpy() for i in range(enc_batch.shape[0])]
word_fixation_sequence['bendr_EEG_enc'] = enc_batch_unpacked

# contextualize
with torch.no_grad():
    context_batch = bendr_contextualizer(enc_batch)
context_batch_unpacked = [context_batch[i,:].numpy() for i in range(context_batch.shape[0])]
word_fixation_sequence['bendr_EEG_context'] = context_batch_unpacked
# %%
