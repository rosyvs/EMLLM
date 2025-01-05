#%%
import pandas as pd
import os


# datawrangling that takes 5 minutes in python and half a day in matlab 



#%% fixations
fix_reportfile = '../info/FixationReport_14feb2023_typosfixed.csv'
dff = pd.read_csv(fix_reportfile, low_memory=False, keep_default_na=True, na_values=['.'])
dff['pID'] = dff['RECORDING_SESSION_LABEL'].str.extract(r'(EML1_\d{3})')
pIDs_in_fixfile = dff['pID'].unique()

#%% saccades
sac_reportfile = '../info/SaccadeReport_with_sen_tag.csv'
dfs = pd.read_csv(sac_reportfile, keep_default_na=True, na_values=['.'], low_memory=False)
dfs['pID'] = dfs['RECORDING_SESSION_LABEL'].str.extract(r'(EML1_\d{3})')
pIDs_in_sacfile = dfs['pID'].unique()

#%%
dfs.rename(columns={'NEXT_FIX_INDEX':'CURRENT_FIX_INDEX','NEXT_FIX_INTEREST_AREA_ID':'CURRENT_FIX_INTEREST_AREA_ID'}, inplace=True)
ID_vars = ['pID','RECORDING_SESSION_LABEL','identifier','CURRENT_FIX_INDEX','CURRENT_FIX_INTEREST_AREA_ID']
sac_vars = ['CURRENT_SAC_AMPLITUDE','CURRENT_SAC_ANGLE','CURRENT_SAC_AVG_VELOCITY', 'CURRENT_SAC_DURATION']
# rename sac_vars 
for v in sac_vars:
    dfs.rename(columns={v:v.replace('CURRENT','INBOUND')}, inplace=True)
    sac_vars[sac_vars.index(v)] = v.replace('CURRENT','INBOUND')
dfs = dfs[ID_vars+sac_vars]

pIDs_not_in_sacfile = [p for p in pIDs_in_fixfile if p not in pIDs_in_sacfile]
pIDs_not_in_fixfile = [p for p in pIDs_in_sacfile if p not in pIDs_in_fixfile]
print(f'pIDs in fixfile but not in sacfile: {pIDs_not_in_sacfile}')
print(f'pIDs in sacfile but not in fixfile: {pIDs_not_in_fixfile}')

# %% 
# convert int to string, missing values to '.'
dff['CURRENT_FIX_INDEX'] = dff['CURRENT_FIX_INDEX'].fillna('.').apply(lambda x: str(int(x)) if x != '.' else x)
dfs['CURRENT_FIX_INDEX'] = dfs['CURRENT_FIX_INDEX'].fillna('.').apply(lambda x: str(int(x)) if x != '.' else x)
dff['CURRENT_FIX_INTEREST_AREA_ID'] = dff['CURRENT_FIX_INTEREST_AREA_ID'].fillna('.').fillna('.').apply(lambda x: str(int(x)) if x != '.' else x)
dfs['CURRENT_FIX_INTEREST_AREA_ID'] = dfs['CURRENT_FIX_INTEREST_AREA_ID'].fillna('.').apply(lambda x: str(int(x)) if x != '.' else x)

#%% match fixations to incoming saccades
#whee identifier oclumns are not unique, enuemrtae them by group
# where there are multiple rows per ID_vars we should take them in order
# we can then merge on the resulting index
dff['order']=dff.fillna('.').groupby(ID_vars).cumcount()
dfs['order']=dfs.fillna('.').groupby(ID_vars).cumcount()

#%%
df = dff.merge(dfs, on=ID_vars+['order'], how='left').drop(columns='order')

# %% save
df.to_csv('../info/FixationReport+InboundSaccades.csv', index=False)

# %%
