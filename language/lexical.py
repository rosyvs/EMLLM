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


def remove_punc(text):
    text=str(text)
    return text.translate(str.maketrans('', '', string.punctuation))

def strip_punc(text):
    text=str(text)
    # strip external punc only, keeping hyphens and apostrophes inside words 
    return text.strip(string.punctuation)

def agg_suprisal_to_words(res):
    # join tokens to get words & sum token surprisals and check these match original words
    # rejoin any tokens starting with Ġ to previous token
    wrd = ''
    surp = 0
    words_from_res = []
    word_surprisals = []
    for r in res:
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

def remove_spaces_before_punc(text):
    return re.sub(r'\s([.,!?;:])', r'\1', text)

def wordlist_to_text(wordlist):
    text = remove_spaces_before_punc(' '.join(wordlist))
    return text

def get_word_surprisals(textlist, model):
    text = wordlist_to_text(textlist)
    this_res = model.surprise(remove_spaces_before_punc(text))[0]
    words_from_res, word_surprisals = agg_suprisal_to_words(this_res)
    return words_from_res, word_surprisals

def match_word_indices(list1, list2):
    # get the index in list1 of the matching words in derivative list2
    # also get the indices of the derivative list2 that have a match in list1
    # so the mapping will be monotonic but not necessarily 1:1
    ix_list1 = []
    ix_list2 = []
    for i, w in enumerate(list2):
        w = strip_punc(w)
        ix = ix_list1[-1] if len(ix_list1) > 0 else 0 # constrain search to be after last match
        while (ix < len(list1)-1) and strip_punc(list1[ix]) != w:
            ix+=1
        if w == strip_punc(list1[ix]):
            ix_list1.append(ix)
            ix_list2.append(i)
    return ix_list1, ix_list2
# general function for lookup surprisal values given text, tokenizer, model
def get_surprisal(text, model):
    this_res = model.surprise(text)[0]
    words_from_res, word_surprisals = agg_suprisal_to_words(this_res)
    return words_from_res, word_surprisals

def add_surprisal_col(df, identifier_col, model, tokenizer, surprisal_col, word_col='IA_LABEL'):
    max_length = tokenizer.model_max_length
    print(f'Adding surprisal values to {surprisal_col} column')
    # initialize column w NA    
    col_to_add = pd.Series([pd.NA]*len(df), index=df.index)
    for text in tqdm(df[identifier_col].unique().tolist()):
        this_textlist = df.loc[df[identifier_col]==text][word_col].values
        this_text = wordlist_to_text(this_textlist)
        this_text_tokens = tokenizer(this_text)['input_ids']
        if len(this_text_tokens) > max_length:
            print(f'{text} is too long, can only model first {max_length} tokens out of {len(this_text_tokens)}')
            continue
        this_res = model.surprise(this_text)[0]
        words_from_res, word_surprisals = agg_suprisal_to_words(this_res)
        ix_text, ix_res = match_word_indices(this_textlist, words_from_res)
        # use the indices to fill the surprisal values
        # get indices rel to this group in full df
        ix_text_full = df.loc[df[identifier_col]==text].index[ix_text].tolist()
        res_to_df = [word_surprisals[i] for i in ix_res]
        if len(ix_text_full) != len(res_to_df):
            raise ValueError(f"Length mismatch: {len(ix_text_full)} indices and {len(res_to_df)} surprisal values")
        col_to_add.iloc[ix_text_full] = res_to_df
    df[surprisal_col] = col_to_add
    return df
