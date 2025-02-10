import os
from zuco_data import load_mat_file, sentence_to_fix_seq, get_min_max_sentenceData
from get_bendr_feats import init_pretrained_bendr_encoder, decimate_and_interpolate_eeg, scale_and_add_scale_channel
import torch
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import mne
from functools import partial

def load_zuco_bendr_channels():
    # detect if current dir is top level of repo or in subfolder eeg already
    if os.path.exists('ZuCo_channels_1020.csv'):
        zuco_chan_df = pd.read_csv('ZuCo_channels_1020.csv')
    else:
        zuco_chan_df = pd.read_csv('eeg/ZuCo_channels_1020.csv')
    return zuco_chan_df

def get_bendr_feats_sent(sent_data, encoder, contextualizer, use_chan_ix, scaler=None, downsampler=None):
    # get BENDR for entire sentence "raw EEG" data
    # returns diict with conv encoded data and contextualized encoded data
    if scaler is None:
        scaler = partial(scale_and_add_scale_channel, dataset_min=-1, dataset_max=1)
    if downsampler is None:
        downsampler = partial(decimate_and_interpolate_eeg, fs_old=500, fs_new=256)
    sent_EEG = sent_data.rawData
    if isinstance(sent_EEG, float):
        print('No EEG data for this sentence (just a float value)')
        return {'bendr_EEG_enc': [], 'bendr_EEG_context':[]}
    sent_EEG = sent_EEG[use_chan_ix,:]
    sent_EEG = scaler(downsampler(sent_EEG))
    sent_EEG_batch = sent_EEG
    if len(sent_EEG_batch) == 0:
        print('No EEG data for this sentence')
        return {'bendr_EEG_enc': [], 'bendr_EEG_context': []}
    with torch.no_grad():
        sent_enc = bendr_encoder(torch.tensor(sent_EEG_batch).unsqueeze(0).float())
        sent_context = bendr_contextualizer(sent_enc)
    print(f'sent_enc shape: {sent_enc.shape}')
    return sent_enc.squeeze().numpy(), sent_context.squeeze().numpy()

def get_bendr_feats_fix_seq(fixations, encoder, contextualizer, use_chan_ix, scaler=None, downsampler=None):
    if scaler is None:
        scaler = partial(scale_and_add_scale_channel, dataset_min=-1, dataset_max=1)
    if downsampler is None:
        downsampler = partial(decimate_and_interpolate_eeg, fs_old=500, fs_new=256)
    eeg_batch = []
    # print(f'len fixations seq: {len(fixations["EEG"])}')
    max_len = 0
    fix_ix = []
    for i,eeg in enumerate(fixations['EEG']):
        if isinstance(eeg, float):
            print(f'No EEG data for fixation {i} (just a float value)')
            continue
        eeg = eeg[use_chan_ix,:]
        eeg = scaler(downsampler(eeg))
        eeg_batch.append(eeg)
        fix_ix.append(i)
        max_len = eeg.shape[1] if eeg.shape[1] > max_len else max_len
    # pad to max length of eegs
    eeg_batch = [torch.nn.functional.pad(torch.tensor(eeg), (0,max_len-eeg.shape[1])) for eeg in eeg_batch]
    if len(eeg_batch) == 0:
        return {'bendr_EEG_enc': [], 'bendr_EEG_context': []}
    # make batch of dims n_fixations x n_channels x n_timepoints
    eeg_batch = torch.stack(eeg_batch).float()
    # print(f'EEG batch shape: {eeg_batch.shape}')

    with torch.no_grad():
        enc_batch = bendr_encoder(eeg_batch)
        # print(f'enc_batch shape: {enc_batch.shape}')
    # unpack the encs into a list of fixaations
    enc_batch_unpacked = [enc_batch[i,:,:].numpy() for i in range(enc_batch.shape[0])]
    # reinsert empty array for fixations without EEG data
    for i in range(len(fixations['EEG'])):
        if i not in fix_ix:
            enc_batch_unpacked.insert(i, None)
    # print(f'enc_batch_unpacked shape: {enc_batch_unpacked[0].shape}')
    with torch.no_grad():
        context_batch = bendr_contextualizer(enc_batch)
    print(f'fixseq context_batch shape: {context_batch.shape}')
    context_batch_unpacked = [context_batch[i,:,:].numpy() for i in range(context_batch.shape[0])]
    for i in range(len(fixations['EEG'])):
        if i not in fix_ix:
            context_batch_unpacked.insert(i, None)
    return enc_batch_unpacked, context_batch_unpacked

def bendr_prepro_zuco(data, bendr_encoder, bendr_contextualizer, use_chan_ix, resume_from=None):
    dataset_min, dataset_max = get_min_max_sentenceData(data)
    scaler = partial(scale_and_add_scale_channel, dataset_min=dataset_min, dataset_max=dataset_max)
    sentences = []
    old_fs = 500
    new_fs = 256
    if resume_from is not None:
        print(f'!!! Resuming from sentence {resume_from}')
    for sent_ix in tqdm(range(len(data))):
        if resume_from is not None and sent_ix < resume_from:
            continue
        print(f'\n...Processing sentence {sent_ix}: {data[sent_ix].content}')
        sent_data = data[sent_ix]
        fix_seq = sentence_to_fix_seq(sent_data)
        sent_enc, sent_enc_context = get_bendr_feats_sent(sent_data, bendr_encoder, bendr_contextualizer, use_chan_ix=use_chan_ix)
        fix_seq_enc, fix_seq_enc_context = get_bendr_feats_fix_seq(fix_seq, bendr_encoder, bendr_contextualizer, use_chan_ix=use_chan_ix)
        fix_seq['bendr_EEG_enc'] = fix_seq_enc
        fix_seq['bendr_EEG_enc_context'] = fix_seq_enc_context
        sent={} # dict to hold all data for this sentence so we can pickle instead of save as .mat   
        sent['fixation_sequence'] = fix_seq
        sent['bendr_EEG_enc'] = sent_enc
        sent['bendr_EEG_enc_context'] = sent_enc_context
        sent['sentence_ix'] = sent_ix
        sent['content'] = sent_data.content
        sentences.append(sent)
    return sentences

if __name__ == '__main__':
    path_to_zuco = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EEG-Gaze/ZuCo/osfstorage'
    RESUME_FROM = None # for debug
    # RESUME_FROM=None
    outpath = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EEG-Gaze/ZuCo/BENDR/task1/'
    os.makedirs(outpath, exist_ok=True)
    subdir = 'task1- SR/Matlab files/'
    zuco_filelist = [f for f in os.listdir(os.path.join(path_to_zuco,subdir)) if f.endswith('.mat')]
    zuco_chan_df = load_zuco_bendr_channels()
    use_chan_ix = zuco_chan_df['zuco_chan_ix'].values

    bendr_encoder, bendr_contextualizer = init_pretrained_bendr_encoder()
    for f in zuco_filelist:
        print(f'Processing {f}')
        data = load_mat_file(os.path.join(path_to_zuco,subdir,f))['sentenceData']

        print(f'...n sentences: {len(data)}') # n sentence
        sentences_bent = bendr_prepro_zuco(data, bendr_encoder, bendr_contextualizer, use_chan_ix, resume_from=RESUME_FROM)
        # save the processed data in a python compatible format
        sentences_bent_file = os.path.join(outpath, f.replace('.mat', '_bendr.pkl'))
        with open(sentences_bent_file, 'wb') as f:
            pickle.dump(sentences_bent, f)
        print(f'Processed data saved to {sentences_bent_file}')


