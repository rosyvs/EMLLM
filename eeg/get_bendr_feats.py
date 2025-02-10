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
from torch.nn.functional import interpolate

# from dn3.configuratron import ExperimentConfig
# from dn3.transforms.instance import To1020
# from dn3.transforms.batch import RandomTemporalCrop
from torch.utils.data import ConcatDataset
# coe to extract features using bendr pretrained weights
# see https://github.com/SPOClab-ca/BENDR
# pretrained weights can be dled from https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha
global PATH_TO_WEIGHTS
PATH_TO_WEIGHTS = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EEG-Gaze/BENDR/checkpoints'
encoder_weights = os.path.join(PATH_TO_WEIGHTS, 'encoder.pt')
context_weights = os.path.join(PATH_TO_WEIGHTS, 'contextualizer.pt')



# config as in pretrained 
args={}
args['hidden_size'] = 512
args['layer_drop'] = 0
args['enc_channels'] = 20  # for 19 EEG + 1 relative amplitude channel, see paper
args['srate_in'] = 256
args['finetuning'] = False
args['dropout'] = 0
# eeg_channels = 8
# sfreq = 1000

# encoder = ConvEncoderBENDR(args.enc_channels, encoder_h=args.hidden_size)
# contextualizer = BENDRContextualizer(encoder.encoder_h, layer_drop=args.layer_drop)
# encoder.load(encoder_weights)
# contextualizer.load(context_weights)

def init_pretrained_bendr_encoder(path_to_weights=PATH_TO_WEIGHTS, args=None):
    if args is None:
        args = {}
        args['hidden_size'] = 512
        args['layer_drop'] = 0
        args['enc_channels'] = 20  # for 19 EEG + 1 relative amplitude channel, see paper
        args['srate_in'] = 256
        args['finetuning'] = False
        args['dropout'] = 0
    encoder_weights = os.path.join(path_to_weights, 'encoder.pt')
    context_weights = os.path.join(path_to_weights, 'contextualizer.pt')
    encoder = ConvEncoderBENDR(args.get('enc_channels',20), encoder_h=args.get('hidden_size',512))
    contextualizer = BENDRContextualizer(encoder.encoder_h, layer_drop=args.get('layer_drop',0.01))
    encoder.load(encoder_weights)
    contextualizer.load(context_weights)
    return encoder, contextualizer

# BENDR channels: get names for hte 19 channels used


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

def decimate_eeg(eeg, decim=2):
    # downsample EEG data from original sampling rate to new sampling rate
    # eeg: EEG data to downsample, shape n_channels x n_timepoints
    # decim: factor by which to downsample
    eeg = eeg[:,::decim]
    return eeg

def downsample_eeg(eeg, fs_old=500, fs_new=256):
    # downsapme for noninteger multiples
    # apply anti-aliasing filter before downsampling
    # eeg: EEG data to downsample, shape n_channels x n_timepoints
    # fs_old: original sampling rate
    # fs_new: new sampling rate
    eeg = mne.io.RawArray(eeg, mne.create_info(ch_names=[f'{ci}' for ci in range(eeg.shape[0])], sfreq=fs_old))
    eeg.resample(fs_new)
    eeg = eeg.get_data()
    return eeg

def decimate_and_interpolate_eeg(eeg, fs_old, fs_new):
    # downsample EEG data from original sampling rate to new sampling rate using method from BENDR training
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
    eeg = decimate_eeg(eeg, decim=decim)
    new_len = int(round(orig_len/fs_old*fs_new))
    eeg_new = interpolate(torch.tensor(eeg).unsqueeze(0), size= new_len, mode='nearest').squeeze().numpy()
    return eeg_new