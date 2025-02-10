# %%
from scipy import io
import h5py
import mat73
import mne 
import os
import torch
import pandas as pd
import numpy as np
from get_bendr_feats import init_pretrained_bendr_encoder
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


