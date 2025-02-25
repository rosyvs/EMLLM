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
from lexical import strip_punc, agg_suprisal_to_words, remove_spaces_before_punc, wordlist_to_text, get_word_surprisals, match_word_indices, add_surprisal_col
# %%
# We load texts according to the IA mapping derived from fixation reports, rather than from the materials directly.
# This is to make indexing onto IAs easier and get surprisal for each word
ia_label_mapping = pd.read_csv('../info/ia_label_mapping.csv')  
ia_label_mapping['IA_ID'] = ia_label_mapping['IA_ID'].astype(int)
ia_label_mapping['text'] =  ia_label_mapping['identifier'].str.replace(r'[0-9]', '', regex=True)
texts = ia_label_mapping['text'].unique().tolist()
pages = ia_label_mapping['identifier'].unique().tolist()

# %% add tag for punctuation
ia_label_mapping['punctuation'] = ia_label_mapping['IA_LABEL'].apply(lambda x: x in ['.', ',', '!', '?', ':', ';', '(', ')', '"', "'"])


# %% Word position in sentence
text = ia_label_mapping['IA_LABEL'].tolist()
# count sentences per identifier
ia_label_mapping['sentence_ix'] = -1
ia_label_mapping['word_in_sentence'] = -1
sentence_ix = 0
word_in_sentence = -1
print('Counting word positions in sentences')
for i, row in tqdm(ia_label_mapping.iterrows(), total=len(ia_label_mapping)):
    if row[ 'punctuation']:
        if word_in_sentence == -1:
            # skip orphaned punc that sometimes occurred after end of sentence. Wait for next word to increment counters
            continue
        ia_label_mapping.loc[i, 'word_in_sentence'] = word_in_sentence
        ia_label_mapping.loc[i, 'sentence_ix'] = sentence_ix
        if ia_label_mapping.loc[i-1,'IA_LABEL'] != 'Mr' and ia_label_mapping.loc[i,'IA_LABEL'] in ['.', '!', '?']:
            # reset sentence
            sentence_ix += 1
            word_in_sentence = -1
    else:
        word_in_sentence += 1
        ia_label_mapping.loc[i, 'word_in_sentence'] = word_in_sentence
        ia_label_mapping.loc[i, 'sentence_ix'] = sentence_ix

# drop orphan punc rows
ia_label_mapping = ia_label_mapping[~ia_label_mapping['word_in_sentence'].eq(-1)]
# add column for sentence-level word counts
sentences = ia_label_mapping['sentence_ix'].unique().tolist()
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

#%% [markdown]
# # surprisal
# - GPT2 has been shown to be the best choice for modeling human surprise in reading in several papers, particularly the 'small' variant is better than larger
# - OPT is a more recent model that is equivalent to GPT3 

#%% Load GPT2 & get surprisal for each word
m = AutoHuggingFaceModel.from_pretrained('gpt2') # 125m
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2') 
ia_label_mapping = add_surprisal_col(ia_label_mapping, 'identifier', m, tokenizer, 'gpt2_surprisal_page')
# add_surprisal_col(ia_label_mapping, 'text', m, tokenizer, 'gpt2_surprisal_wholetext')

#%% opt is more like GPT3 and can model longer context
m = AutoHuggingFaceModel.from_pretrained('facebook/opt-125m', model_class='gpt')
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
ia_label_mapping = add_surprisal_col(ia_label_mapping, 'identifier', m, tokenizer, 'opt-125m_surprisal_page')
ia_label_mapping = add_surprisal_col(ia_label_mapping, 'text', m, tokenizer, 'opt-125m_surprisal_wholetext')


# # display words from one page (identifier) and color code by surprisal
# page = ia_label_mapping.loc[ia_label_mapping['identifier']=='Bias0']
# # sort by IA_ID
# plt.barh(page['IA_ID'], page['opt-125m_surprisal_page'])
# plt.yticks(page['IA_ID'], page['IA_LABEL'])
# plt.xlabel('Surprisal')
# plt.title('Surprisal values for each word on one page')
# plt.gca().invert_yaxis()
# plt.show()

#%% compare surprisal values from GPT2 and OPT
ia_label_mapping['gpt2_surprisal_page'].corr(ia_label_mapping['opt-125m_surprisal_page'])
ia_label_mapping['opt-125m_surprisal_page'].corr(ia_label_mapping['opt-125m_surprisal_wholetext'])
# visualize surprisal values
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
plt.figure(figsize=(10,20))
plt.subplot(2,1,1)
sns.scatterplot(data=ia_label_mapping, x='gpt2_surprisal_page', y='opt-125m_surprisal_page')
plt.title('Surprisal values')
plt.subplot(2,1,2)
sns.scatterplot(data=ia_label_mapping, x='opt-125m_surprisal_wholetext', y='opt-125m_surprisal_page')
ia_label_mapping['gpt2_surprisal_page'].hist(bins=100)
#%% final save
ia_label_mapping.to_csv('../info/ia_label_mapping_opt_surprisal.csv', index=False)   
ia_label_mapping= pd.read_csv('../info/ia_label_mapping_opt_surprisal.csv')

