import pandas as pd
import numpy as np
import re

# look in overall resutls folder and compare skip_reasons.csv between analysis verisons

dir_out_par = '/Volumes/Blue1TB/EEG_processed/'
pat = 'FRP_TRF_lexical_'
# get all folders in dir_out_par
version_dirs = [d for d in os.listdir(dir_out_par) if os.path.isdir(os.path.join(dir_out_par, d)) and pat in d]

# iterate over these and read in skip_reasons for each, make a new df of skip reasons for all with a column for each version

skip_reasons_all=[]
for version in version_dirs:
    skip_reasons_file = os.path.join(dir_out_par, version, 'skip_reasons.csv')
    verstr = version.replace(pat,'')
    if os.path.exists(skip_reasons_file):
        df = pd.read_csv(skip_reasons_file, header=None, names=['pID', 'skip_reason'])
        # set pID as index
        df.set_index('pID', inplace=True)
        # rename skip reason to version name
        df.rename(columns={'skip_reason': verstr}, inplace=True)

        # get list of actual processsed files (*-evk.fif) and add these to the df with skip reason 'Processed'
        # get all files in dir_out_par
        processed_files = [f for f in os.listdir(os.path.join(dir_out_par, version)) if f.endswith('-evk.fif')]
        # get pIDs from filenames (formatted like EML1_\d{3})
        pIDs = [re.search(r'EML1_\d{3}', f).group(0) for f in processed_files]
        # add these as rows to the df with skip reason 'Processed'
        df_ = pd.DataFrame(pIDs, columns=['pID'])
        df_[verstr] = 'Processed'
        # set pID as index
        df_.set_index('pID', inplace=True)
        # concat with df
        df = pd.concat([df, df_])
        # sort
        df.sort_index(inplace=True)
        # drop duplicates (keep last)
        df.drop_duplicates(inplace=True, keep='last')
        # add to list
        skip_reasons_all.append(df)
    
# concatenate the dfs and keep pID as index
skip_reasons_all = pd.concat(skip_reasons_all, axis=1)
# sort by pID
skip_reasons_all.sort_index(inplace=True)
# put pID as normal column
skip_reasons_all.reset_index(inplace=True)


# %% lsit all pIDs wehre skip_reason = 'Processed' in v7 but not in v2d
todos = skip_reasons_all.loc[skip_reasons_v7=='Processed','pID']
