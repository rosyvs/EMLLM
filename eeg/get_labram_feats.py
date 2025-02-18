from braindecode.models.labram import Labram
import numpy as np
import torch
import torch.nn as nn

# you might need to install braindecode from github for latest models
# pip install -U https://api.github.com/repos/braindecode/braindecode/zipball/master#egg=braindecode

# %% randomly initialized? 
tokeniser = Labram(n_times=500, n_chans = 105, neural_tokenizer=True, n_outputs=2)
eeg = torch.Tensor(np.random.randn(1,105, 500))
with torch.no_grad():
    tok = tokeniser(eeg)

#%%
import sys
sys.path.append('..')
from ext.LaBraM.modeling_finetune import labram_base_patch200_200
encoder = labram_base_patch200_200(pretrained=True)
from ext.LaBraM.run_class_finetuning import get_models, create_model
pretrained_model = create_model(
    'labram_base_patch200_200', 
    checkpoint_path='/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EEG-Gaze/LaBraM/checkpoints/labram-base.pth',)
# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
checkpoint = torch.load('/Users/roso8920/Emotive Computing Dropbox/Rosy Southwell/EEG-Gaze/LaBraM/checkpoints/labram-base.pth',
map_location=device)

# from https://github.com/BINE022/EEGPT/blob/main/downstream/linear_probe_LaBraM_PhysioP300.py:
new_checkpoint = {}
for k,v in checkpoint['model'].items():
    if k.startswith('student.'):
        new_checkpoint[k[len('student.'):]] = v
model = create_model("labram_base_patch200_200", 
                        # checkpoint_path= ,
                        qkv_bias=False,
                        rel_pos_bias=True,
                        num_classes=4,
                        drop_rate=0.0,
                        drop_path_rate=0.1,
                        attn_drop_rate=0.0,
                        drop_block_rate=None,
                        use_mean_pooling=True,
                        init_scale=0.001,
                        use_rel_pos_bias=True,
                        use_abs_pos_emb=True,
                        init_values=0.1,)
model.load_state_dict(new_checkpoint, strict=False)
model.to(device)
model.eval()
model(eeg)
# %%
