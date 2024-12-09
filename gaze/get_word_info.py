#%%
import os
import pandas as pd
import re
import numpy as np
import pandas as pd
#%% Fixation labels 
# - one row per fixation for all participants
# - interest area indices given where available. THese correspond to word/pinc bonding boxes
# - metadata for aligning with other files:
#   - fixation index
#   - fixation duration (for a sanity check)
#   - CURRENT_FIX_START (timestamp of fix start - seems to start from 0 )
# fix_labelfile = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/Processed/Medha_models/FixationReport24Sept23.txt'
# df = pd.read_csv(fix_labelfile, sep='\t', na_values='.')
# df.head(20)
# df.columns
# # unclear whether we should use "INDEX" or "ID"
# # ID" seems to be INDEX but sometimes + 1 or 2
# # df['CURRENT_FIX_INTEREST_AREA_INDEX'] = df['CURRENT_FIX_INTEREST_AREA_INDEX'].fillna(-1)
# (df['CURRENT_FIX_INTEREST_AREA_INDEX'].unique().astype(int))
# (df['CURRENT_FIX_INTEREST_AREA_ID'].unique().astype(int))
# df['IA_ID_INDEX_diff'   ] = df['CURRENT_FIX_INTEREST_AREA_ID'] - df['CURRENT_FIX_INTEREST_AREA_INDEX']
# df['IA_ID_INDEX_diff'].unique()
# df['identifier'].unique()
# 
# %%  how does this differ from various other fixation report?
fix_reportfile = '/Volumes/Blue1TB/EyeMindLink/DataViewer/DataViewer_EML1/Output/FixationReport_14feb2023.txt'
df = pd.read_csv(fix_reportfile, low_memory=False, sep='\t')
df.head(20)
df.columns
df['identifier'].unique()
# fill '.' as NA for INDEX And ID columns
df['CURRENT_FIX_INTEREST_AREA_INDEX'] = df['CURRENT_FIX_INTEREST_AREA_INDEX'].replace('.', np.nan)
df['CURRENT_FIX_INTEREST_AREA_ID'] = df['CURRENT_FIX_INTEREST_AREA_ID'].replace('.', np.nan)
df = df[df['EYE_USED']=='RIGHT']
df['IA_ID_INDEX_diff'   ] = pd.to_numeric(df['CURRENT_FIX_INTEREST_AREA_ID'], errors='coerce')- pd.to_numeric(df['CURRENT_FIX_INTEREST_AREA_INDEX'], errors='coerce')
df['IA_ID_INDEX_diff'].unique()
df['CURRENT_FIX_INTEREST_AREA_LABEL'].unique()
ia_label_mapping = df[['identifier','CURRENT_FIX_INTEREST_AREA_ID', 'CURRENT_FIX_INTEREST_AREA_LABEL']].drop_duplicates()
ia_label_mapping = ia_label_mapping.rename(columns={'CURRENT_FIX_INTEREST_AREA_ID':'IA_ID', 'CURRENT_FIX_INTEREST_AREA_LABEL':'IA_LABEL'})
ia_label_mapping['IA_ID'] = pd.to_numeric(ia_label_mapping['IA_ID'], errors='coerce', downcast='integer')
ia_label_mapping = ia_label_mapping[~ia_label_mapping['identifier'].str.contains('Sham')]
ia_label_mapping = ia_label_mapping.dropna(subset='IA_ID').sort_values(by=['identifier', 'IA_ID']).reset_index(drop=True)

# #%% some IAs are mistyped as these were entered manually by an RA, but are corect in the actual text.
# # load original materials 
# text_file = '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/Experiment/Materials/Texts_utf8.txt'
# # thsio is tab seprated
# texts_orig = pd.read_csv(text_file, sep='\t')
# texts_orig['identifier'] = texts_orig['TextID']+(texts_orig['Page']-1).astype(str)
# # split text into words and make a new df with row for each word and puncutation item (not splitting hyphenated words)
# # keep the punctuation as separate words
# texts_orig['Text2'] = texts_orig['Text'].str.replace(r'([.,!?])', r' \1', regex=True)
# texts_orig['words'] = texts_orig['Text2'].str.split(' ')
# # strip punc of fwords, but keep sole punctuation items


# texts_orig_words = texts_orig.explode('words')
# # greedy strip punctuation
# punctuation = ['.', ',', '!', '?', "'", '"', '(', ')',':']
# texts_orig_words['words'] = texts_orig_words['words'].apply(lambda x: x.strip(''.join(punctuation)) if x not in punctuation else x)

# # try to align words in IA mapping to words in original text
# texts_orig_words['index'] = texts_orig_words.groupby(['identifier'])['words'].cumcount()
# ia_label_mapping['index'] = ia_label_mapping.groupby(['identifier'])['IA_LABEL'].cumcount()
# # merge on index
# ia_label_mapping = ia_label_mapping.merge(texts_orig_words, on=['identifier','index'], how='left')

#%% fix typos in IA labels
typos_fixed={
    'maipulations':'manipulations',
    'hypothsis':'hypothesis',
    'sistuations':'situations',
    'paticipant':'participant',
    'intruments':'instruments',
    'qualitics':'qualities',
}
for typo, fix in typos_fixed.items():
    ia_label_mapping.loc[ia_label_mapping['IA_LABEL']==typo, 'IA_LABEL'] = fix

# fix in the fixation report too
for typo, fix in typos_fixed.items():
    df.loc[df['CURRENT_FIX_INTEREST_AREA_LABEL']==typo, 'CURRENT_FIX_INTEREST_AREA_LABEL'] = fix


#%%
ia_label_mapping.to_csv('../info/ia_label_mapping.csv', index=False)   
df.to_csv('../info/FixationReport_14feb2023_typosfixed.csv', index=False)

# # %% compare df and df to check IA columns match
# df_merged = pd.merge(df[df['identifier']=='Bias6'], df[df['identifier']=='Bias6'], how='left',on=['identifier','TRIAL_INDEX','CURRENT_FIX_INDEX'], suffixes=('_medha', '_dataviewer'))
# # sort columsn to make comparison easier
# df_merged = df_merged.reindex(sorted(df_merged.columns), axis=1)
# df_merged.head(20)


# #%% are there reparsed files with blinks and saccades? Not sure if realignment would affect these
# msgdf = pd.read_csv('/Volumes/Blue1TB/EyeMindLink/DataViewer/DataViewer_EML1/Output/MessageReport_181_allvars_16Jan2023.txt', sep='\t', low_memory=False, na_values='.')

# # %% IA-word mapping appears to be here:
# # '/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/EML_AOI'
# df = pd.read_csv('/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/DataViewer/DataViewer_EML1/Output/IA_Report_withwords_MC.txt',sep='\t',  low_memory=False)
# df.head(20)
# ia_label_mapping = df[['identifier','IA_ID', 'IA_LABEL']].drop_duplicates()
# ia_label_mapping = ia_label_mapping[~ia_label_mapping['identifier'].str.contains('Sham')].sort_values(by=['identifier', 'IA_ID']).reset_index(drop=True)
# ia_label_mapping.to_csv('../info/ia_label_mapping.csv', index=False)   

# # %% Load materials (text)
# # extract texts from ia mapping
# texts = ia_label_mapping['identifier'].unique().tolist()
# for text in texts:
#     this_text = ia_label_mapping.loc[ia_label_mapping['identifier']==text]['IA_LABEL'].values
# texts_orig = pd.read_csv('/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EyeMindLink/Experiment/Materials/Texts.csv')


# %%
