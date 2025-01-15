import mne
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import meegkit
from mne.datasets import sample
from mne.stats.regression import linear_regression_raw
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import scipy


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

