#%%
import os
import pandas as pd
import re
import numpy as np

#%% Fixation labels 
# - one row per fixation for all participants
# - interest area indices given where available. THese correspond to word/pinc bonding boxes
# - metadata for aligning with other files:
#   - fixation index
#   - fixation duration (for a sanity check)
#   - CURRENT_FIX_START (timestamp of fix start - seems to start from 0 )
fix_labelfile = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/Processed/Medha_models/FixationReport24Sept23.txt'
df = pd.read_csv(fix_labelfile, sep='\t', na_values='.')
df.head(20)

#%%
# unclear whether we should use "INDEX" or "ID"
# ID" seems to be INDEX but sometimes + 1 or 2
# df['CURRENT_FIX_INTEREST_AREA_INDEX'] = df['CURRENT_FIX_INTEREST_AREA_INDEX'].fillna(-1)
(df['CURRENT_FIX_INTEREST_AREA_INDEX'].unique().astype(int))
(df['CURRENT_FIX_INTEREST_AREA_ID'].unique().astype(int))
df['IA_ID_INDEX_diff'   ] = df['CURRENT_FIX_INTEREST_AREA_ID'] - df['CURRENT_FIX_INTEREST_AREA_INDEX']
df['IA_ID_INDEX_diff'].unique()
# h
# %%  how does this differ from various other fixation report?
fix_reportfile = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/Processed/Data Viewer Trial Reports/FixationReport_01Sept2021.csv'
df2 = pd.read_csv(fix_reportfile)
df2.head(20)
df2.columns
# A: it doesnt have any IA columns

# %% IA-word mapping appears to be here:
# '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/EML_AOI'
df_ia = pd.read_csv('/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/DataViewer/DataViewer_EML1/Output/IA_Report_withwords_MC.txt',sep='\t',  low_memory=False)
df_ia.head(20)
ia_label_mapping = df_ia[['identifier','IA_ID', 'IA_LABEL']].drop_duplicates()
ia_label_mapping = ia_label_mapping[~ia_label_mapping['identifier'].str.contains('Sham')].sort_values(by=['identifier', 'IA_ID']).reset_index(drop=True)
ia_label_mapping.to_csv('../info/ia_label_mapping.csv', index=False)   

# %% Load materials (text)
# extract texts from ia mapping
texts = ia_label_mapping['identifier'].unique().tolist()
for text in texts:
    this_text = ia_label_mapping.loc[ia_label_mapping['identifier']==text]['IA_LABEL'].values
texts_orig = pd.read_csv('/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/Experiment/Materials/Texts.csv')
# %% Load surprisal for text 
# surprisal model: OPT
import transformers
from transformers import AutoTokenizer, OPTModel
from surprisal import AutoHuggingFaceModel
import torch
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model = OPTModel.from_pretrained("facebook/opt-125m")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)


#%%
m = AutoHuggingFaceModel.from_pretrained('facebook/opt-125m', model_class='gpt')

sentences=['my dog is cute. That is why I love him', 
'my dog is cute. That is why I love soup',
'I have no teeth. That is why I love him', 
'I have no teeth. That is why I love soup']

for result in m.surprise(sentences):
    print(result)

# %% Get surprisal for each text
texts = ia_label_mapping['identifier'].unique().tolist()
for text in texts:
    this_text = ia_label_mapping.loc[ia_label_mapping['identifier']==text]['IA_LABEL'].values
    this_res = m.surprise(' '.join(this_text))[0]
    print(f'{len(this_res)} surprisal values from {len(this_text)} words')
# there is not a 1:1 mapping from words to surprisal values.

#%%
# TODO: use GPT tokenizer to get IA mapping for each token 
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
tokens = tokenizer(' '.join(this_text), return_tensors='pt')
# convert tokens back to words to check against original words
words_from_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
len(tokens['input_ids'][0])
print(f'{len(this_text)} tokenized words -> {len(this_res)} surprisal values')
# still not same number of words and surprisal values

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
words_from_res
# check if this_text and words_from_res match
print(f'{len(this_text)} words -> {len(words_from_res)} words')
for i, (t, w, s) in enumerate(zip(this_text, words_from_res, word_surprisals)):
    if t != w:
        print(f'{i}: {t} != {w}')
    else:
        print(f'{i}: {t} == {w}, surprisal: {s}')



# %% TODO: use levenshtein distance to match words to surprisal output

