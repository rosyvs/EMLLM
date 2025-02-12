from braindecode.models.labram import Labram
import numpy as np
import torch
import torch.nn as nn

# you might need to install braindecode from github for latest models
# pip install -U https://api.github.com/repos/braindecode/braindecode/zipball/master#egg=braindecode

tokeniser = Labram(n_times=500, n_chans = 105, neural_tokenizer=True, n_outputs=2)
eeg = torch.Tensor(np.random.randn(1,105, 500))
with torch.no_grad():
    tok = tokeniser(eeg)

#%%
import sys
sys.path.append('..')
from ext.LaBraM.modeling_finetune import labram_base_patch200_200
encoder = labram_base_patch200_200(pretrained=True)
from ext.LaBraM.run_class_finetuning import get_models
# %%
