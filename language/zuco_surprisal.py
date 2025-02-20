# %% Load surprisal for text 
# surprisal model: OPT
import os
import pandas as pd
import re
import numpy as np
import transformers
from transformers import AutoTokenizer, OPTModel, GPT2TokenizerFast
from surprisal import AutoHuggingFaceModel
import torch
from tqdm import tqdm
import nltk
from tqdm import tqdm
import string
import sys
sys.path.append('..')
from eeg.zuco_data import load_mat_file
from lexical import strip_punc, remove_punc, agg_suprisal_to_words, remove_spaces_before_punc, wordlist_to_text, get_word_surprisals, match_word_indices, add_surprisal_col

# %%
# We load texts according to the IA mapping derived from fixation reports, rather than from the materials directly.
# This is to make indexing onto IAs easier and get surprisal for each word
path_to_zuco = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EEG-Gaze/ZuCo/osfstorage'
subdir = 'task1- SR/Preprocessed/'
sentences = load_mat_file(os.path.join(path_to_zuco, subdir, 'sentencesSR.mat'))['sentences']
# save to txt
with open(os.path.join(path_to_zuco, subdir, 'sentencesSR.txt'), 'w') as f:
    for s in sentences:
        f.write(s + '\n')
sentences = pd.DataFrame({"Text": sentences})
sentences["sentence_ix"] = sentences.index
sentences['Text2'] = sentences['Text'].str.replace(r'([.,!?])', r' \1', regex=True)
sentences['IA_LABEL'] = sentences['Text2'].str.split(' ')
# strip punc of fwords, but keep sole punctuation items
ia_label_mapping = sentences.explode('IA_LABEL') # IA_LABEL includes punctuation
ia_label_mapping['IA_INDEX'] = ia_label_mapping.groupby('sentence_ix').cumcount()
ia_label_mapping['punctuation'] = ia_label_mapping['IA_LABEL'].apply(lambda x: x in ['.', ',', '...','--','_','!', '?', ':', ';', '(', ')', '"', "'"])
# split on hyphens and explode again
ia_label_mapping['word'] = ia_label_mapping['IA_LABEL'].str.split('-').str.split('--')
ia_label_mapping = ia_label_mapping.explode('word')
ia_label_mapping['word'] = ia_label_mapping['word'].apply(strip_punc)
ia_label_mapping['identifier'] = ia_label_mapping['sentence_ix'].apply(lambda x: f'sentence{x:03d}')
ia_label_mapping.drop_duplicates(inplace=True)
# drop rows with emoty string or na in word
ia_label_mapping = ia_label_mapping[~ia_label_mapping['word'].eq('')].dropna(subset=['word'])
# %% Word position in sentence
# Initialize word position in sentence
ia_label_mapping['word_in_sentence'] = -1
prev_sentence_ix = -1
word_in_sent_col = []
print('Counting word positions in sentences')
for i, row in tqdm(ia_label_mapping.iterrows(), total=len(ia_label_mapping)):
    if row['sentence_ix'] != prev_sentence_ix:
        # reset word position in sentence
        word_in_sentence = 0
        prev_sentence_ix = row['sentence_ix']
        word_in_sent_col.append(word_in_sentence)
    else:
        if not row['punctuation']:
            word_in_sentence += 1
        word_in_sent_col.append(word_in_sentence)
ia_label_mapping['word_in_sentence'] = word_in_sent_col

# drop orphan punc rows
ia_label_mapping = ia_label_mapping[~ia_label_mapping['word_in_sentence'].eq(-1)]
ia_label_mapping.reset_index(drop=True, inplace=True)
# group by sentence_ix and get max word_in_sentence for word count in sentence
ia_label_mapping['sentence_word_count'] = ia_label_mapping.groupby('sentence_ix')['word_in_sentence'].transform('max')
ia_label_mapping['relative_word_position'] = ia_label_mapping['word_in_sentence'] / (ia_label_mapping['sentence_word_count'])
# %% frequency
from wordfreq import word_frequency
ia_label_mapping['word_freq'] = ia_label_mapping['IA_LABEL'].apply(lambda x: word_frequency(x, 'en'))

# %% is function word
from nltk.corpus import stopwords
def is_function_word(word):
    stop_words = set(stopwords.words('english'))
    return word.lower() in stop_words
ia_label_mapping['stop_word'] = ia_label_mapping['IA_LABEL'].apply(is_function_word)

#%% Load GPT2 & get surprisal for each word
m = AutoHuggingFaceModel.from_pretrained('gpt2') # 125m
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2') 
ia_label_mapping = add_surprisal_col(ia_label_mapping, 'sentence_ix', m, tokenizer, 'gpt2_surprisal', word_col = 'word')



#%%
# df = ia_label_mapping
# identifier_col = 'sentence_ix'
# model = m
# tokenizer = tokenizer
# surprisal_col = 'gpt2_surprisal'
# word_col = 'word'
# max_length = tokenizer.model_max_length
# print(f'Adding surprisal values to {surprisal_col} column')
# # initialize column w NA    
# col_to_add = pd.Series([pd.NA]*len(df), index=df.index)
# for text in tqdm(df[identifier_col].unique().tolist()):
#     this_textlist = df.loc[df[identifier_col]==text][word_col].values
#     this_text = wordlist_to_text(this_textlist)
#     this_text_tokens = tokenizer(this_text)['input_ids']
#     if len(this_text_tokens) > max_length:
#         print(f'{text} is too long, can only model first {max_length} tokens out of {len(this_text_tokens)}')
#         continue
#     this_res = model.surprise(this_text)[0]
#     words_from_res, word_surprisals = agg_suprisal_to_words(this_res)
#     ix_text, ix_res = match_word_indices(this_textlist, words_from_res)
#     # use the indices to fill the surprisal values
#     # get indices rel to this group in full df
#     ix_text_full = df.loc[df[identifier_col]==text].index[ix_text].tolist()
#     res_to_df = [word_surprisals[i] for i in ix_res]
#     if len(ix_text_full) != len(res_to_df):
#         raise ValueError(f"Length mismatch: {len(ix_text_full)} indices and {len(res_to_df)} surprisal values")
#     col_to_add.iloc[ix_text_full] = res_to_df
# df[surprisal_col] = col_to_add

#%% final save
ia_label_mapping.to_csv('../info/zuco_gpt_surprisal.csv', index=False)   



# %%
