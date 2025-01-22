import mne
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import meegkit
from mne.datasets import sample
from mne.stats.regression import linear_regression_raw, linear_regression, _clean_rerp_input, _make_evokeds
from mne._fiff.pick import _picks_to_idx, pick_info, pick_types
from mne.epochs import BaseEpochs
from mne.evoked import Evoked, EvokedArray
from mne.source_estimate import SourceEstimate
from mne.utils import _reject_data_segments, fill_doc, logger, warn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import scipy
from scipy import stats, sparse, linalg

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

def predict_EEG(X, betas):
    return np.dot(X, betas)

def ridge_model(X, y, solver='auto', alpha=1):
    res = Ridge(solver=solver,alpha=alpha).fit(X, y)
    return res

def ridge_stats(model, X, y):
    # estimate t stat and p values for betas from ridge regression
    # https://stats.stackexchange.com/questions/326294/how-to-calculate-t-statistics-for-ridge-regression

    # Calculate the mean squared error (mse) of the residuals.
    # Calculate the inverse of the X.T @ X matrix (XTX_inv).
    # Calculate standard errors (se) for each coefficient.
    # Calculate t-statistics (t_stats) for each coefficient.
    # Calculate p-values:
    # Use the stats.t.cdf function to calculate the cumulative distribution function of the t-distribution.
    # Calculate p-values (p_values) based on the t-statistics.
    n = X.shape[0]
    p = X.shape[1]
    beta = model.coef_
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - p)
    XTX_inv = np.linalg.inv(np.dot(X.T, X).toarray())
    se = np.sqrt(np.diagonal(mse * XTX_inv)) 
    t_stats = beta / se

    # Calculate p-values
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p))

    # Create a dataframe for results
    stats_dict={
        'n_times': n,
        'n_predictors': p,
        'y_pred':y_pred,
        'betas': beta,
        't-stats': t_stats,
        'mse': mse,
        'p-values': p_values
    }
    return stats_dict



########
# my edits of linear_regression_raw to return the model design matrix used


def ridge_regression_raw(
    raw,
    events,
    event_id=None,
    tmin=-0.1,
    tmax=1,
    covariates=None,
    reject=None,
    flat=None,
    tstep=1.0,
    decim=1,
    picks=None,
    model="ridge",
):
    """Estimate regression-based evoked potentials/fields by linear modeling.

    This models the full M/EEG time course, including correction for
    overlapping potentials and allowing for continuous/scalar predictors.
    Internally, this constructs a predictor matrix X of size
    n_samples * (n_conds * window length), solving the linear system
    ``Y = bX`` and returning ``b`` as evoked-like time series split by
    condition. See :footcite:`SmithKutas2015`.

    Parameters
    ----------
    raw : instance of Raw
        A raw object. Note: be very careful about data that is not
        downsampled, as the resulting matrices can be enormous and easily
        overload your computer. Typically, 100 Hz sampling rate is
        appropriate - or using the decim keyword (see below).
    events : ndarray of int, shape (n_events, 3)
        An array where the first column corresponds to samples in raw
        and the last to integer codes in event_id.
    event_id : dict | None
        As in Epochs; a dictionary where the values may be integers or
        iterables of integers, corresponding to the 3rd column of
        events, and the keys are condition names.
        If None, uses all events in the events array.
    tmin : float | dict
        If float, gives the lower limit (in seconds) for the time window for
        which all event types' effects are estimated. If a dict, can be used to
        specify time windows for specific event types: keys correspond to keys
        in event_id and/or covariates; for missing values, the default (-.1) is
        used.
    tmax : float | dict
        If float, gives the upper limit (in seconds) for the time window for
        which all event types' effects are estimated. If a dict, can be used to
        specify time windows for specific event types: keys correspond to keys
        in event_id and/or covariates; for missing values, the default (1.) is
        used.
    covariates : dict-like | None
        If dict-like (e.g., a pandas DataFrame), values have to be array-like
        and of the same length as the rows in ``events``. Keys correspond
        to additional event types/conditions to be estimated and are matched
        with the time points given by the first column of ``events``. If
        None, only binary events (from event_id) are used.
    reject : None | dict
        For cleaning raw data before the regression is performed: set up
        rejection parameters based on peak-to-peak amplitude in continuously
        selected subepochs. If None, no rejection is done.
        If dict, keys are types ('grad' | 'mag' | 'eeg' | 'eog' | 'ecg')
        and values are the maximal peak-to-peak values to select rejected
        epochs, e.g.::

            reject = dict(grad=4000e-12, # T / m (gradiometers)
                          mag=4e-11, # T (magnetometers)
                          eeg=40e-5, # V (EEG channels)
                          eog=250e-5 # V (EOG channels))

    flat : None | dict
        For cleaning raw data before the regression is performed: set up
        rejection parameters based on flatness of the signal. If None, no
        rejection is done. If a dict, keys are ('grad' | 'mag' |
        'eeg' | 'eog' | 'ecg') and values are minimal peak-to-peak values to
        select rejected epochs.
    tstep : float
        Length of windows for peak-to-peak detection for raw data cleaning.
    decim : int
        Decimate by choosing only a subsample of data points. Highly
        recommended for data recorded at high sampling frequencies, as
        otherwise huge intermediate matrices have to be created and inverted.
    %(picks_good_data)s
    model : str | callable
    sklearn model

    Returns
    -------
    evokeds : dict
        A dict where the keys correspond to conditions and the values are
        Evoked objects with the ER[F/P]s. These can be used exactly like any
        other Evoked object, including e.g. plotting or statistics.
    X: design matrix timeexpanded
    stats : dict
        beta, stderr, t_val, p_val, mlog10_p_val

    References
    ----------
    .. footbibliography::
    """
    if isinstance(model, str):
        if model not in {"ridge"}:
            raise ValueError(f"No such solver: {model}")
        if model == "ridge":
            print("Using Ridge regression model with defaul parameters (alpha=1)")
            model = ridge_model
    elif callable(model):
        pass
    else:
        raise TypeError("The solver must be a str or a callable.")

    # build data
    data, info, events = _prepare_rerp_data(raw, events, picks=picks, decim=decim)

    if event_id is None:
        event_id = {str(v): v for v in set(events[:, 2])}

    # build predictors
    X, conds, cond_length, tmin_s, tmax_s = _prepare_rerp_preds(
        n_samples=data.shape[1],
        sfreq=info["sfreq"],
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        covariates=covariates,
    )

    # remove "empty" and contaminated data points
    X, data = _clean_rerp_input(X, data, reject, flat, decim, info, tstep)

    # solve linear system
    fitted = model(X, data.T)
    coefs = fitted.coef_
    if len(coefs.shape)==1:
        coefs = np.expand_dims(coefs, axis=0)
    if coefs.shape[0] != data.shape[0]:
        raise ValueError(
            f"solver output has unexcepted shape {coefs.shape}. Supply a "
            "function that returns coefficients in the form "
            "(n_targets, n_features), where "
            f"n_targets == n_channels == {data.shape[0]}."
        )

    # construct Evoked objects to be returned from output
    evokeds, regressor_indices = _make_evokeds(coefs, conds, cond_length, tmin_s, tmax_s, info)

    # get stats #TODO: actually compute these
    stats = ridge_stats(fitted, X, data.T)

    return X, regressor_indices, evokeds, stats


def _prepare_rerp_data(raw, events, picks=None, decim=1):
    """Prepare events and data, primarily for `linear_regression_raw`."""
    picks = _picks_to_idx(raw.info, picks)
    info = pick_info(raw.info, picks)
    decim = int(decim)
    with info._unlock():
        info["sfreq"] /= decim
    data, times = raw[:]
    data = data[picks, ::decim]
    if len(set(events[:, 0])) < len(events[:, 0]):
        raise ValueError(
            "`events` contains duplicate time points. Make "
            "sure all entries in the first column of `events` "
            "are unique."
        )

    events = events.copy()
    events[:, 0] -= raw.first_samp
    events[:, 0] //= decim
    if len(set(events[:, 0])) < len(events[:, 0]):
        raise ValueError(
            "After decimating, `events` contains duplicate time "
            "points. This means some events are too closely "
            "spaced for the requested decimation factor. Choose "
            "different events, drop close events, or choose a "
            "different decimation factor."
        )

    return data, info, events


def _prepare_rerp_preds(
    n_samples, sfreq, events, event_id=None, tmin=-0.1, tmax=1, covariates=None
):
    """Build predictor matrix and metadata (e.g. condition time windows)."""
    conds = list(event_id)
    if covariates is not None:
        conds += list(covariates)

    # time windows (per event type) are converted to sample points from times
    # int(round()) to be safe and match Epochs constructor behavior
    if isinstance(tmin, float | int):
        tmin_s = {cond: int(round(tmin * sfreq)) for cond in conds}
    else:
        tmin_s = {cond: int(round(tmin.get(cond, -0.1) * sfreq)) for cond in conds}
    if isinstance(tmax, float | int):
        tmax_s = {cond: int(round(tmax * sfreq) + 1) for cond in conds}
    else:
        tmax_s = {cond: int(round(tmax.get(cond, 1.0) * sfreq)) + 1 for cond in conds}

    # Construct predictor matrix
    # We do this by creating one array per event type, shape (lags, samples)
    # (where lags depends on tmin/tmax and can be different for different
    # event types). Columns correspond to predictors, predictors correspond to
    # time lags. Thus, each array is mostly sparse, with one diagonal of 1s
    # per event (for binary predictors).

    cond_length = dict()
    xs = []
    for cond in conds:
        tmin_, tmax_ = tmin_s[cond], tmax_s[cond]
        n_lags = int(tmax_ - tmin_)  # width of matrix
        if cond in event_id:  # for binary predictors
            ids = (
                [event_id[cond]] if isinstance(event_id[cond], int) else event_id[cond]
            )
            onsets = -(events[np.isin(events[:, 2], ids), 0] + tmin_)
            values = np.ones((len(onsets), n_lags))

        else:  # for predictors from covariates, e.g. continuous ones
            covs = covariates[cond]
            if len(covs) != len(events):
                error = (
                    f"Condition {cond} from ``covariates`` is not the same length as "
                    "``events``"
                )
                raise ValueError(error)
            onsets = -(events[np.where(covs != 0), 0] + tmin_)[0]
            v = np.asarray(covs)[np.nonzero(covs)].astype(float)
            values = np.ones((len(onsets), n_lags)) * v[:, np.newaxis]

        cond_length[cond] = len(onsets)
        xs.append(sparse.dia_matrix((values, onsets), shape=(n_samples, n_lags)))

    return sparse.hstack(xs), conds, cond_length, tmin_s, tmax_s


def _make_evokeds(coefs, conds, cond_length, tmin_s, tmax_s, info):
    """Create a dictionary of Evoked objects.

    These will be created from a coefs matrix and condition durations.
    """
    evokeds = dict()
    cumul = 0
    regressor_indices = {}
    for cond in conds:
        tmin_, tmax_ = tmin_s[cond], tmax_s[cond]
        evokeds[cond] = EvokedArray(
            coefs[:, cumul : cumul + tmax_ - tmin_],
            info=info,
            comment=cond,
            tmin=tmin_ / float(info["sfreq"]),
            nave=cond_length[cond],
            kind="average",
        )  # nave and kind are technically incorrect #TODO: umm? 
        regressor_indices[cond] = (cumul, cumul + tmax_ - tmin_)
        cumul += tmax_ - tmin_
    return evokeds, regressor_indices

def _evokeds_to_coefs(evokeds, regressor_indices):
    ncoefs = max(regressor_indices.values())[1]
    nchans = evokeds[list(evokeds.keys())[0]].data.shape[0]
    coefs = np.zeros((nchans, ncoefs))
    # use regressor_indices to get the correct order of the betas
    for cond in evokeds.keys():
        start, end = regressor_indices[cond]
        coefs[:,start:end] = evokeds[cond].data
    return coefs