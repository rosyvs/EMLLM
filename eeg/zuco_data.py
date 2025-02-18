# %%
from scipy import io
import h5py
import mat73
import mne 
import os
import torch
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from eeg.get_bendr_feats import init_pretrained_bendr_encoder
bendr_encoder, bendr_contextualizer = init_pretrained_bendr_encoder()

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


def sentence_to_fix_seq(sent):
    # get fixation sequence from sentence structure, i.e. ordered by time not word
    fix_seq = []
    # check this sentence has words
    if isinstance(sent.word, float):
        print('No words in this sentence')
        return {'content': sent.content, 'fixations': [], 'EEG': []}
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

def get_min_max_sentenceData(data):
    # get min and max values of entier sentenceData EEG
    min_overall = 0
    max_overall = 0
    for s in data:
        min_val = np.min(s.rawData)
        max_val = np.max(s.rawData)
        if min_val < min_overall:
            min_overall = min_val
        if max_val > max_overall:
            max_overall = max_val
    return min_overall, max_overall


#%% TOOLS FOR ZUCO RAW DATA
import re
from mne.io.eeglab.eeglab import _check_load_mat, RawEEGLAB, _read_annotations_eeglab

# order by the digit before _EEG or _ET
def get_block_no(f):
    # pattern is integer before _EEG or _ET but there can be other _ in the filename
    ans = re.search(r'.*(\d+).*_E[ET].*', f)
    return int(ans.group(1)) if ans else None

def get_fix_df(et):
    cols = et['eyeevent'].fixations.colheader
    data = et['eyeevent'].fixations.data
    df = pd.DataFrame(data, columns=cols)
    df['latency'] = df['latency'].astype(int)

    return df
def label_fixations_with_event(et):
    # get fixations
    fix_df = get_fix_df(et)
    # get events
    et_events = pd.DataFrame(et['event'], columns=['latency','event'])
    # et_events contains onsets of events, label all fixatoins with latency of event they follow
    fix_df['event'] = np.nan
    for i, fix in fix_df.iterrows():
        # get event after this fixation
        event = et_events[et_events['latency'] > fix['latency']].iloc[0]
        fix_df.at[i,'event'] = event['event'].astype(int)
    return fix_df

def mne_from_zucoeeg(eeg, event_dict=None):
    # convert zuco eeg to mne
    # eeg: zuco eeg data
    # returns mne raw object
    eeg = _check_load_mat(os.path.join(path_to_zuco,subdir,pID, eeg_file), uint16_codec=None)
    sfreq = eeg.srate
    print('sfreq:', sfreq)
    times = eeg.times
    ch_names = eeg.chanlocs['labels']
    ch_types = ['eeg']*len(ch_names)
    # sfreq = 500
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg.data, info)
    # get events
    events = pd.DataFrame(eeg.event)
    events['value'] = events['type'].astype(int) # IDK why this is just 'trigger'
    # get description as a string if event_dict is not None
    if event_dict is not None:
        events['type'] = events['value'].map(event_dict)
    annot = mne.Annotations(onset=np.array(events['latency'])/sfreq, duration=np.array(events['duration'])/sfreq, description=events['type'])
    raw.set_annotations(annot)
    return raw, events

def get_event_dict(ver = 1, task=1):
    events_v1_task1 =  {10: 'sentence_onset',
    11: 'sentence_offset',
    12: 'control_onset',
    13: 'control_offset_question_onset',
    15: 'question_answered',
    100:'start',
    102: 'start',
    18:'end',
    22: 'end'}
    if ver == 1 and task == 1: 
        return events_v1_task1
    else:
        raise NotImplementedError('Only version 1 task 1 is implemented')