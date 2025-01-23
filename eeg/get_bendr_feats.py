import os
import numpy as np
import pandas as pd
from bendr import ConvEncoderBENDR, BENDRContextualizer, BendingCollegeWav2Vec
import torch
import tqdm
import argparse
from scipy import io
import h5py
import mat73
from mne.io import read_raw_eeglab
# from dn3.configuratron import ExperimentConfig
# from dn3.transforms.instance import To1020
# from dn3.transforms.batch import RandomTemporalCrop
from torch.utils.data import ConcatDataset
def load_mat_file(file_name):
    try:
        data = io.loadmat(file_name, squeeze_me=True, struct_as_record=False)['sentenceData']
    except OSError:
        data = mat73.loadmat(file_name)
    print(f'loaded data of type {type(data)}, length {len(data)}')
    return data
# coe to extract features using bendr pretrained weights
# see https://github.com/SPOClab-ca/BENDR
# pretrained weights can be dled from https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha

path_to_weights = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EEG-Gaze/BENDR/checkpoints'

encoder_weights = os.path.join(path_to_weights, 'encoder.pt')
context_weights = os.path.join(path_to_weights, 'contextualizer.pt')
args = argparse.Namespace()

# config as in pretrained 
args.hidden_size = 512
args.layer_drop = 0.01 # thats what is in the pretraining yaml in BENDR repo
args.enc_channels = 20  # for 19 EEG + 1 relative amplitude channel, see paper
args.srate_in = 256

eeg_channels = 8
sfreq = 1000

encoder = ConvEncoderBENDR(args.enc_channels, encoder_h=args.hidden_size)
contextualizer = BENDRContextualizer(encoder.encoder_h, layer_drop=args.layer_drop)
encoder.load(encoder_weights)
contextualizer.load(context_weights)

# BENDR channels: get names for hte 19 channels used


# select these channels from EEG data (missing channels set to 0)