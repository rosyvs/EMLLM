from scipy import io
import h5py
import mat73
from mne.io import read_raw_eeglab
import os

# load some example EEG data
# ZuCo
path_to_zuco = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EEG-Gaze/ZuCo/osfstorage'
file_name = os.path.join(path_to_zuco,"task1- SR/Matlab files/resultsZAB_SR.mat")
# data = io.loadmat(file_name, squeeze_me=True, struct_as_record=False)
data = load_mat_file(file_name)

# browse fields
print(len(data)) # n sentences
sent_ix = 0
print(data[sent_ix]._fieldnames)
sent = data[sent_ix]
# raw EEG
sent_EEG = sent.rawData
sent_EEG.shape # dim: n channels x n timepoints
print(sent.content)
print(sent.word[0]._fieldnames)
print(sent.word[0].fixPositions) # indices of fixations for this word
print(sent.word[0].nFixations) # indices of fixations for this word
print(sent.word[0].rawET[0]) # n fix. * timepoints? Some are 1d, others have 2d with first dim=4


print(len(data[0].word))
word_data = data[0].word

# example: get first word
print(word_data[0].content)
print(word_data[0].rawEEG[0].shape) # len rawEEG = n fixations on word. Each rawEEG is chan*timepoints. variable length

# list fixation timestamps and use to get EEG for each fixation


# EEG for an example fixation
egeeg = word_data[0].rawEEG[0]
# channels names

# randomly select 20 channels (hack)
egeeg = egeeg[:20]
with torch.no_grad():
    enc = encoder(torch.tensor(egeeg).unsqueeze(0).float())


# EEG file in EEGLab format
eegfile = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EEG-Gaze/ZuCo/osfstorage/task3 - TSR/Preprocessed/ZAB/gip_ZAB_TSR1_EEG.mat'
eeg = read_raw_eeglab(eegfile)
# get channels names
