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
args['layer_drop'] = 0.01
args['enc_channels'] = 20  # for 19 EEG + 1 relative amplitude channel, see paper
args['srate_in'] = 256

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
        args['layer_drop'] = 0.01
        args['enc_channels'] = 20  # for 19 EEG + 1 relative amplitude channel, see paper
        args['srate_in'] = 256
    encoder_weights = os.path.join(path_to_weights, 'encoder.pt')
    context_weights = os.path.join(path_to_weights, 'contextualizer.pt')
    encoder = ConvEncoderBENDR(args.get('enc_channels',20), encoder_h=args.get('hidden_size',512))
    contextualizer = BENDRContextualizer(encoder.encoder_h, layer_drop=args.get('layer_drop',0.01))
    encoder.load(encoder_weights)
    contextualizer.load(context_weights)
    return encoder, contextualizer

# BENDR channels: get names for hte 19 channels used


# scale channel
    #     if add_scale_ind:
    #         if dataset.info is None or dataset.info.data_max is None or dataset.info.data_min is None:
    #             # print("Can't add scale index with dataset that is missing info.")
    #             pass
    #         else:
    #             self.max_scale = dataset.info.data_max - dataset.info.data_min
    #     self.return_mask = return_mask

    # def __call__(self, x):
    #     if self.max_scale is not None:
    #         scale = 2 * (torch.clamp_max((x.max() - x.min()) / self.max_scale, 1.0) - 0.5)
    #     else:
    #         scale = 0

    #     x = (x.transpose(1, 0) @ self.mapping).transpose(1, 0)

    #     for ch_type_inds in (EEG_INDS, EOG_INDS, REF_INDS, EXTRA_INDS):
    #         x[ch_type_inds, :] = min_max_normalize(x[ch_type_inds, :])

    #     used_channel_mask = self.mapping.sum(dim=0).bool()
    #     x[~used_channel_mask, :] = 0

    #     x[SCALE_IND, :] = scale

    #     if self.return_mask:
    #         return (x, used_channel_mask)
    #     else:
    #         return x

# select these channels from EEG data (missing channels set to 0)