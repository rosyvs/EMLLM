# %% Load surprisal for text 
# surprisal model: OPT
import os
import pandas as pd
import re
import numpy as np
import transformers
from transformers import AutoTokenizer, OPTModel
from surprisal import AutoHuggingFaceModel
import torch
from tqdm import tqdm
#%% load models
# load model using surprisal wrapper
m = AutoHuggingFaceModel.from_pretrained('facebook/opt-125m', model_class='gpt')
# sentences=['my dog is cute. That is why I love him', 
# 'my dog is cute. That is why I love soup',
# 'I have no teeth. That is why I love him', 
# 'I have no teeth. That is why I love soup']
# for result in m.surprise(sentences):
#     print(result)

#%% join tokens to get words & sum token surprisals and check these match original words
def agg_suprisal_to_words(res):
    # rejoin any tokens starting with Ġ to previous token
    wrd = ''
    surp = 0
    words_from_res = []
    word_surprisals = []
    for r in this_res:
        if r[0] == '</s>':
            continue
        if r[0].startswith('Ġ'):
            words_from_res.append(wrd)
            word_surprisals.append(surp)
            wrd = r[0][1:]
            surp = r[1]
        else:
            wrd += r[0]
            surp += r[1]
    words_from_res.append(wrd)
    word_surprisals.append(surp)
    return words_from_res, word_surprisals


# %% Get surprisal for each text
# We load texts according to the IA mapping derived from fixation reports, rather than from the materials directly.
# This is to make indexing onto IAs easier and get surprisal for each word
ia_label_mapping = pd.read_csv('../info/ia_label_mapping.csv')  
ia_label_mapping['text'] =  ia_label_mapping['identifier'].str.replace(r'[0-9]', '', regex=True)
# remove number from identifier to get text

texts = ia_label_mapping['text'].unique().tolist()
texts

for text in tqdm(texts):
    this_text = ia_label_mapping.loc[ia_label_mapping['text']==text]['IA_LABEL'].values
    this_res = m.surprise(' '.join(this_text))[0]
    # print(f'{len(this_res)} surprisal values from {len(this_text)} words')

    words_from_res, word_surprisals = agg_suprisal_to_words(this_res)
    assert(all(words_from_res == this_text))
    ia_label_mapping.loc[ia_label_mapping['text']==text, 'opt-125m_surprisal_wholetext'] = word_surprisals
    # check if this_text and words_from_res match
    # print(f'{len(this_text)} words -> {len(words_from_res)} words')
    # for i, (t, w, s) in enumerate(zip(this_text, words_from_res, word_surprisals)):
    #     if t != w:
    #         print(f'{i}: {t} != {w}')
        # else:
        #     print(f'{t}: {s}')

# %% Get surprisal for each page (not using context of whole text)
pages = ia_label_mapping['identifier'].unique().tolist()


for text in tqdm(pages):
    this_text = ia_label_mapping.loc[ia_label_mapping['identifier']==text]['IA_LABEL'].values
    this_res = m.surprise(' '.join(this_text))[0]

    words_from_res, word_surprisals = agg_suprisal_to_words(this_res)
    assert(all(words_from_res == this_text))
    ia_label_mapping.loc[ia_label_mapping['identifier']==text, 'opt-125m_surprisal_page'] = word_surprisals

#%% compare text and page surprisal
ia_label_mapping['surprisal_diff'] = ia_label_mapping['opt-125m_surprisal_wholetext'] - ia_label_mapping['opt-125m_surprisal_page']
# top 10 differences
ia_label_mapping.sort_values(by='surprisal_diff', ascending=False).head(10)

#%% save to file
ia_label_mapping.to_csv('../info/ia_label_mapping_opt_surprisal.csv', index=False)
#%%
# NOTE: didn't work
# use GPT tokenizer to get IA mapping for each token 
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
# model = OPTModel.from_pretrained("facebook/opt-125m")
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
# tokens = tokenizer(' '.join(this_text), return_tensors='pt')
# # convert tokens back to words to check against original words
# words_from_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
# len(tokens['input_ids'][0])
# print(f'{len(this_text)} tokenized words -> {len(this_res)} surprisal values')
# # still not same number of words and surprisal values







# %% TODO: use levenshtein distance to match words to surprisal output

