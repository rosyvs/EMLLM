# %%
import scipy
import h5py
import mat73
import mne
import os
import torch
import pandas as pd
import numpy as np

def load_mat_file(file_name):
    try:
        data = scipy.io.loadmat(file_name, squeeze_me=True, struct_as_record=False)
        print(f'Loaded data using scipy.io.loadmat')
    except Exception as e:
        print(f'Failed to load using scipy.io.loadmat: {e}')
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
# data = scipy.io.loadmat(file_name, squeeze_me=True, struct_as_record=False)
data = load_mat_file(file_name)['sentenceData']

# %% browse fields
print(f'n sentences: {len(data)}') # n sentences
sent_ix = 0
sent = data[sent_ix]
print('sentence fieldnames:')
print(sent._fieldnames)
# raw EEG
sent_EEG = sent.rawData
print(f'sentence EEG rawData shape:{sent_EEG.shape}') # dim: n channels x n timepoints
print(sent.content)
print('word fieldnames:')
print(sent.word[0]._fieldnames)
print(sent.word[0].fixPositions) # indices of fixations for this word
print(f'n fixations on word:{sent.word[0].nFixations}') # count fixations per word
print('raw ET data shape for 1 fixation on this word:', sent.word[0].rawET[0].shape)
print('word level raw EEG shape for 1 fixation on this word:', sent.word[0].rawEEG[0].shape)

def sentence_to_fix_seq(data, sent_ix):
    sent = data[sent_ix]
    # get fixation sequence from sentence structure, i.e. ordered by time not word
    fix_seq = []
    for i, word in enumerate(sent.word):
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
    # append EEG to fixation sequence
    fix_EEG = []
    for i, row in fixations.iterrows():
        fix_EEG.append(sent.word[row['word_ix']].rawEEG[row['count_on_word']])
    # append eyetracker data to fixation sequence
    fix_ET = []
    for i, row in fixations.iterrows():
        fix_ET.append(sent.word[row['word_ix']].rawET[row['count_on_word']])
    word_fixation_sequence = {'content': sent.content, 'fixations': fixations, 'EEG': fix_EEG}
    return word_fixation_sequence

# %% get fixation sequence from sentence structure, i.e. ordered by time not word
fix_seq = []
for i, word in enumerate(sent.word):
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
# append EEG to fixation sequence
fix_EEG = []
for i, row in fixations.iterrows():
    fix_EEG.append(sent.word[row['word_ix']].rawEEG[row['count_on_word']])
# append eyetracker data to fixation sequence
fix_ET = []
for i, row in fixations.iterrows():
    fix_ET.append(sent.word[row['word_ix']].rawET[row['count_on_word']])
word_fixation_sequence = {'content': sent.content, 'fixations': fixations, 'EEG': fix_EEG}

# %%
from torch.nn.functional import interpolate


def decimate_and_interpolate_zuco(eeg, fs_old, fs_new):
    # downsample EEG data from original sampling rate to new sampling rate
    # eeg: EEG data to downsample, shape n_channels x n_timepoints
    # fs_old: original sampling rate
    # fs_new: new sampling rate
    # from BENDR: "Specifically, we over- or undersampled (by whole multiples, for lower and higher sampling frequencies, respectfully [sic]) 
    # to get nearest to the target sampling frequency of 256 Hz. 
    # Then, nearest-neighbor interpolation was used to obtain the precise frequency (as was done in prior work Kostas and Rudzicz, 2020a).""
    # specifically they use nn.functional interpolate with mode='nearest' (see dn3 code)
    # - Rosy
    orig_len = eeg.shape[1]
    decim = fs_old//fs_new # first integer decimation
    eeg = decimate_zuco(eeg, decim=decim)
    new_len = int(round(orig_len/fs_old*fs_new))
    eeg_new = interpolate(torch.tensor(eeg).unsqueeze(0), size= new_len, mode='nearest').squeeze().numpy()
    return eeg_new

eeg = sent.word[0].rawEEG[0]
eeg_new = decimate_and_interpolate_zuco(eeg, fs_old=500, fs_new=256)
# plot both
import matplotlib.pyplot as plt
plt.plot(eeg[0,:])
plt.plot([fs_old/fs_new*i for i in range(len(eeg_new[0,:]))],eeg_new[0,:])
plt.show()
#%%
def get_min_max_sentenceData(data):
    # get min and max values of entier sentenceData EEG
    min_overall = 0
    max_overall = 0
    for s in data:
        min_val = np.min(s.rawData)
        max_val = np.max(eeg)
        if min_val < min_overall:
            min_overall = min_val
        if max_val > max_overall:
            max_overall = max_val
    return min_overall, max_overall

def scale_and_add_scale_channel(eeg, dataset_min, dataset_max):
    # scale data to range -1,1 and add a scale channel which is a constant value for all timepoints:
    # (max(eeg) - min(eeg))/(dataset_max - dataset_min)
    # eeg: EEG data to scale, shape n_channels x n_timepoints
    # dataset_min: minimum value of dataset
    # dataset_max: maximum value of dataset
    # returns scaled data with an additional scale channel in final position
    scale = (np.max(eeg) - np.min(eeg))/(dataset_max - dataset_min)
    eeg = np.concatenate([eeg, np.ones((1,eeg.shape[1]))*scale], axis=0)
    return eeg

dataset_min, dataset_max = get_min_max_sentenceData(data)
eeg = sent.word[0].rawEEG[0]
eeg_scaled = scale_and_add_scale_channel(eeg, dataset_min, dataset_max)

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
channels_1020.to_csv('ZuCo_channels_1020.csv', index=False)
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
    eeg = np.concatenate([eeg, np.zeros((1,eeg.shape[1]))], axis=0)# add additional channel for scale, set to all 0 for now
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

# %% get BENDR for entire sentence
sent_EEG = sent.rawData
sent_EEG = sent_EEG[use_chan_ix,:]
sent_EEG_batch = np.concatenate([sent_EEG, np.zeros((1,sent_EEG.shape[1]))], axis=0)# add additional channel for scale, set to all 0 for now
with torch.no_grad():
    sent_enc = bendr_encoder(torch.tensor(sent_EEG_batch).unsqueeze(0).float())
    sent_context = bendr_contextualizer(sent_enc)

sentence_bendr = {'content': sent.content, 'bendr_EEG_enc': sent_enc.numpy(), 'bendr_EEG_context': sent_context.numpy(),
'word_fixation_sequence': word_fixation_sequence}
sentence_bendr['source_filename'] = file_name   

#%% load a pkl of extracted  features
import pickle
fn = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EEG-Gaze/ZuCo/BENDR/task1/resultsZKB_SR_bendr.pkl'
with open(fn, 'rb') as f:
    data = pickle.load(f)

# %%
