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
# log transom
ia_label_mapping['log_word_freq'] = ia_label_mapping['word_freq'].apply(lambda x: np.log(x) if x > 0 else np.nan)
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

# %% load back in just for plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
ia_label_mapping = pd.read_csv('../info/ia_label_mapping_opt_surprisal.csv')

# get correl between surprisal and word frequency, scatter plot
plt.figure(figsize=(10,5))
sns.scatterplot(data=ia_label_mapping, x='log_word_freq', y='gpt2_surprisal_page')
plt.title('Surprisal values')
plt.xlabel('Log Word Frequency')
plt.ylabel('Surprisal')
# text on plot giving correl
corr = ia_label_mapping['word_freq'].corr(ia_label_mapping['gpt2_surprisal_page'])
plt.text(0.1, 0.9, f'Correlation: {corr:.2f}', fontsize=12, transform=plt.gca().transAxes)
plt.show()

# get correl matrix between surprisal, word frequency, relative word position, sentence word count
# 1. for all words
corr = ia_label_mapping[['gpt2_surprisal_page', 'log_word_freq', 'relative_word_position', 'sentence_word_count']].corr()
plt.figure(figsize=(10,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation matrix: all words')
plt.show()
# 2. for only content words
corr = ia_label_mapping[~ia_label_mapping['stop_word']][['gpt2_surprisal_page', 'log_word_freq', 'relative_word_position', 'sentence_word_count']].corr()
plt.figure(figsize=(10,5))              
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation matrix: content words only')

# %% table of descriptives
descriptives = ia_label_mapping[[
    'gpt2_surprisal_page', 'log_word_freq', 'relative_word_position', 'sentence_word_count']].describe()

# same for content words only
descriptives_content = ia_label_mapping[~ia_label_mapping['stop_word']][['gpt2_surprisal_page', 'log_word_freq', 'relative_word_position', 'sentence_word_count']].describe()

# combine two descriptives into one table
descriptives_combined = pd.concat([descriptives, descriptives_content], axis=1, keys=['All Words', 'Content Words'])

# format to 3dp and export
descriptives_combined = descriptives_combined.round(3)
descriptives_combined.to_csv('../info/ia_label_mapping_opt_surprisal_descriptives.csv')

# get percent of content words 
content_word_count = ia_label_mapping[~ia_label_mapping['stop_word']].shape[0]
all_word_count = ia_label_mapping.shape[0]
pct_content_words = content_word_count / all_word_count * 100
print(f'Percentage of content words: {pct_content_words:.2f}%')

# get percent of content words per page 
pct_content_words_per_page = ia_label_mapping.groupby('identifier')['stop_word'].apply(lambda x: (~x).sum() / x.shape[0] * 100)

# get mean word count per  sentence at page level
mean_sentence_word_count = ia_label_mapping.groupby('identifier')['sentence_word_count'].mean()


# combine into table
descriptives_page = pd.DataFrame({
    'mean_sentence_word_count': [mean_sentence_word_count.mean()],
    'mean_sentence_word_count_sd': [mean_sentence_word_count.std()],
    'pct_content_words': [pct_content_words_per_page.mean()],
    'pct_content_words_sd': [pct_content_words_per_page.std()]
}, index=['overall'])
descriptives_page.to_csv('../info/ia_label_mapping_opt_surprisal_descriptives_page.csv')
 # %%

# plot ia_label_mapping histograms for all words and content words, for the 4 columns
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# set overall title
fig.suptitle('All Words', fontsize=16)
# Plot histograms for all words
sns.histplot(ia_label_mapping['gpt2_surprisal_page'], bins=20, ax=axes[0])
sns.histplot(ia_label_mapping['log_word_freq'], bins=20, ax=axes[1])
sns.histplot(ia_label_mapping['relative_word_position'], bins=20, ax=axes[2])
# Set titles for each subplot
axes[0].set_xlabel('GPT2 Surprisal')
axes[1].set_xlabel('Log Word Frequency')
axes[2].set_xlabel('Relative Word Position')
plt.tight_layout()
fig.savefig('../info/histograms_all_words.png')

# %% same for content words only
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# set overall title
fig.suptitle('Content Words', fontsize=16)
sns.histplot(ia_label_mapping[~ia_label_mapping['stop_word']]['gpt2_surprisal_page'], bins=20, ax=axes[0])
sns.histplot(ia_label_mapping[~ia_label_mapping['stop_word']]['log_word_freq'], bins=20, ax=axes[1])
sns.histplot(ia_label_mapping[~ia_label_mapping['stop_word']]['relative_word_position'], bins=20, ax=axes[2])
# Set titles for each subplot
axes[0].set_xlabel('GPT2 Surprisal')
axes[1].set_xlabel('Log Word Frequency')
axes[2].set_xlabel('Relative Word Position')

plt.tight_layout()
fig.savefig('../info/histograms_content_words.png')
# %%
