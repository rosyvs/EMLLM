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
from functools import partial



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

def ridge_stats(model, X, y, estimate_pvals=False):
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
    if estimate_pvals:
        # try to invert but sometomes X.T X is singular
        try:
            XTX_inv = np.linalg.inv(np.dot(X.T, X))
        except:
            print('Ridge regression stats: X.T X is singular, adding small value to diagonal')
            XTX_inv = np.linalg.inv(np.dot(X.T, X) + 1e-10*np.eye(X.shape[1]))
        se = np.sqrt(np.diagonal(mse * XTX_inv)) 
        t_stats = beta / se

        # Calculate p-values
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p))
    else:
        t_stats = None
        p_values = None

    # Create a dataframe for results
    stats_dict={
        'n_times': n,
        'n_predictors': p,
        'y_pred':y_pred,
        'betas': beta,
        't-stats': t_stats,
        'mse': mse,
        'p-values': p_values,
        'residuals': residuals
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
    estimate_pvals=True,
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

    estimate_pvals : bool
        If True, estimate t-statistics and p-values for the betas (can take a while)

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

    # get stats
    stats = ridge_stats(fitted, X, data.T, estimate_pvals=estimate_pvals)
    stats['regressor_indices'] = regressor_indices

    if estimate_pvals:
        keys = ["betas", "t-stats", "p-values"]
    else:   
        keys = ["betas"]
        # unpack certain stats by condition
    for key in keys:
        stats[key] = {cond: stats[key][:,idx[0]:idx[1]] for cond, idx in regressor_indices.items()}
    return X, evokeds, stats


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

# wrap solver in a function to make it a callable
def ridge_solver(X, y, alpha=1): 
    res = Ridge(solver='auto',alpha=alpha).fit(X, y).coef_ #TODO: what else comes from Ridge
    if len(res.shape)==1:
        res = np.expand_dims(res, axis=0)
    return res
def ridge_model(X, y, solver='auto', alpha=1):
    res = Ridge(solver=solver,alpha=alpha).fit(X, y)
    return res

from scipy.stats import ttest_rel # https://mne.discourse.group/t/mne-stats-permutation-cluster-1samp-test/3530/3
def ttest_rel_nop(*args):
    tvals, _ = ttest_rel(*args)
    return tvals

def comprehension_above_chance(x,n, printout=False):
    pvals = []
    weights=[]
    for n_i in np.unique(n):
        x_i = x[n==n_i]
        p = 1/n_i
        # binomial test
        res = scipy.stats.binomtest(x_i.sum(), n=len(x_i), p=p)
        if res.pvalue > 0.05 and printout:
            print(f'performance on {n_i} alternatives is {x_i.sum()}/{len(x_i)} ({x_i.sum()/len(x_i):.2f}) which is not different from chance (p={res.pvalue:.2f})')
        elif printout:
            print(f'performance on {n_i} alternatives is {x_i.sum()}/{len(x_i)} ({x_i.sum()/len(x_i):.2f}) which is different from chance (p={res.pvalue:.2f})')
        pvals += [res.pvalue]
        weights += [len(x_i)]
    # weighted mean p value
    res = scipy.stats.combine_pvalues(pvals, method='stouffer', weights = weights)
    return pvals, res

# function to detect collinear colum groups in X using qr from scipy
def detect_collinear(X, tol=1e-5):
    """Detect collinear columns in X using QR decomposition."""
    Q, R = scipy.linalg.qr(X, mode='economic')
    # Find the rank of R
    rank = np.sum(np.abs(np.diag(R)) > tol)
    # Find the indices of the linearly dependent columns
    collinear_indices = np.where(np.abs(np.diag(R)) <= tol)[0]
    return collinear_indices

def ix_to_regressor_label(ix, regressor_indices):
    """Convert indices to regressor labels."""
    for key, value in regressor_indices.items():
        if ix >= value[0] and ix < value[1]:
            return key
    return None

def collinear_regressors(X, regressor_indices):
    """Detect collinear regressors in X using QR decomposition."""
    if isinstance(X, scipy.sparse.csr_matrix):
        X = X.toarray()
    # get the indices of the collinear columns
    collinear_indices = detect_collinear(X)
    # get the labels of the collinear regressors
    collinear_regressors = [ix_to_regressor_label(i, regressor_indices) for i in collinear_indices]
    # remove duplicates
    collinear_regressors = list(set(collinear_regressors))
    return collinear_regressors










#### edits to mne plto functions to make nicer looking plots
# plot sig ranges
def plot_with_stats(evk, pvals, channel='CPz', ax=None, alpha=0.05):
    erp_times = evk.times
    condname =  evk.comment
    channels = evk.ch_names
    if isinstance(channel, str):
        chan_ix = channels.index(channel)
    else:
        chan_ix = channel
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(12, 6))
    evk.plot(picks=channels[chan_ix], axes=ax)
    # make bool array for if p<0.05
    sig = pvals[chan_ix] < alpha
    if np.any(sig):
        # consolidate ranges: find changepoints in sig
        changepoints = np.where(np.diff(sig))[0]
        if sig[0]:
            changepoints = np.concatenate([[0], changepoints])
        if sig[-1]:
            changepoints = np.concatenate([changepoints, [len(sig)]])
        assert len(changepoints) % 2 == 0
        # make list of start and end points from successive pairs
        ranges = np.array([[changepoints[i], changepoints[i+1]] for i in range(0,len(changepoints),2)])
        # colour significant effects with shaded background
        for i, p in enumerate(ranges):
            ax.axvspan(erp_times[p[0]], erp_times[p[1]], color='k', alpha=0.2)
    return ax

def plot_cluster(clusters, cluster_p_values, times, ax, yloc=0, add_text=True):  # Removed unused tcfe parameter
    h = None
    for i_c, c in enumerate(clusters):
        pcount = 0
        if cluster_p_values[i_c] <= 0.05:
            pcount+= 1
            h = ax.axvspan(times[c[0]][0], times[c[0]][-1], color="k", alpha=0.2)
            # get max y on axes and use this as yloc
            if pcount % 2 == 0:
                va = 'top'
            else:
                va= 'bottom'
            if add_text:
                ax.text(
                    times[c[0]][0],
                    yloc,
                    f" p = {cluster_p_values[i_c]:.3f}\n {times[c[0]][0]:.3f} – {times[c[0]][-1]:.3f} s",
                    color="k",
                    fontsize=8,
                    verticalalignment=va

                )
    if h:
        ax.legend((h,), ("cluster p-value < 0.05",), loc='lower right')
    # ax.set_xlabel("time (ms)")
    # ax.set_ylabel("stat")
    return ax


from mne.viz.evoked import (_check_time_unit, 
                            _prepare_joint_axes, 
                            _butterfly_on_button_press, 
                            _butterfly_onpick, 
                            _check_spatial_colors,
                            _rgb,
                            _handle_spatial_colors,
                            _add_nave
)
from mne._fiff.pick import (
    _DATA_CH_TYPES_SPLIT,
    _PICK_TYPES_DATA_DICT,
    _VALID_CHANNEL_TYPES,
    _picks_to_idx,
    # channel_indices_by_type,
    channel_type,
    pick_info,
)
from mne.defaults import _handle_default

from mne.utils import (
    _check_ch_locs,
    _check_if_nan,
    # _clean_names,
    # _is_numeric,
    _pl,
    # _to_rgb,
    # _validate_type,
    # fill_doc,
    logger,
    # verbose,
    warn,
)
from mne.viz.topo import _plot_evoked_topo
from mne.viz.topomap import (
    _check_sphere,
    # _draw_outlines,
    # _get_pos_outlines,
    # _make_head_outlines,
    # _prepare_topomap,
    # _prepare_topomap_plot,
    _set_contour_locator,
    # plot_topomap,
)
from mne.viz.utils import (
    # DraggableColorbar,
    _check_cov,
    _check_delayed_ssp,
    _check_option,
    _check_time_unit,
    _draw_proj_checkbox,
    # _get_cmap,
    # _get_color_list,
    # _make_combine_callable,
    # _plot_masked_image,
    _prepare_joint_axes,
    _process_times,
    # _prop_kw,
    # _set_title_multiple_electrodes,
    _set_window_title,
    # _setup_ax_spines,
    # _setup_cmap,
    _setup_plot_projector,
    _setup_vmin_vmax,
    # _triage_rank_sss,
    # _trim_ticks,
    _validate_if_list_of_axes,
    plt_show,
)

def plot_evoked_joint(
    evoked,
    times="peaks",
    title="",
    picks=None,
    exclude=None,
    show=True,
    ts_args=None,
    topomap_args=None,
    cmap=None,
    # linewidth=1
):
    """Plot evoked data as butterfly plot and add topomaps for time points.

    .. note:: Axes to plot in can be passed by the user through ``ts_args`` or
              ``topomap_args``. In that case both ``ts_args`` and
              ``topomap_args`` axes have to be used. Be aware that when the
              axes are provided, their position may be slightly modified.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked instance.
    times : float | array of float | "auto" | "peaks"
        The time point(s) to plot. If ``"auto"``, 5 evenly spaced topographies
        between the first and last time instant will be shown. If ``"peaks"``,
        finds time points automatically by checking for 3 local maxima in
        Global Field Power. Defaults to ``"peaks"``.
    title : str | None
        The title. If ``None``, suppress printing channel type title. If an
        empty string, a default title is created. Defaults to ''. If custom
        axes are passed make sure to set ``title=None``, otherwise some of your
        axes may be removed during placement of the title axis.
    %(picks_all)s
    exclude : None | list of str | 'bads'
        Channels names to exclude from being shown. If ``'bads'``, the
        bad channels are excluded. Defaults to ``None``.
    show : bool
        Show figure if ``True``. Defaults to ``True``.
    ts_args : None | dict
        A dict of ``kwargs`` that are forwarded to :meth:`mne.Evoked.plot` to
        style the butterfly plot. If they are not in this dict, the following
        defaults are passed: ``spatial_colors=True``, ``zorder='std'``.
        ``show`` and ``exclude`` are illegal.
        If ``None``, no customizable arguments will be passed.
        Defaults to ``None``.
    topomap_args : None | dict
        A dict of ``kwargs`` that are forwarded to
        :meth:`mne.Evoked.plot_topomap` to style the topomaps.
        If it is not in this dict, ``outlines='head'`` will be passed.
        ``show``, ``times``, ``colorbar`` are illegal.
        If ``None``, no customizable arguments will be passed.
        Defaults to ``None``.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure | list
        The figure object containing the plot. If ``evoked`` has multiple
        channel types, a list of figures, one for each channel type, is
        returned.

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    from matplotlib.patches import ConnectionPatch

    if ts_args is not None and not isinstance(ts_args, dict):
        raise TypeError(f"ts_args must be dict or None, got type {type(ts_args)}")
    ts_args = dict() if ts_args is None else ts_args.copy()
    ts_args["time_unit"], _ = _check_time_unit(
        ts_args.get("time_unit", "s"), evoked.times
    )
    topomap_args = dict() if topomap_args is None else topomap_args.copy()

    got_axes = False
    illegal_args = {"show", "times", "exclude"}
    for args in (ts_args, topomap_args):
        if any(x in args for x in illegal_args):
            raise ValueError(
                "Don't pass any of {} as *_args.".format(", ".join(list(illegal_args)))
            )
    if ("axes" in ts_args) or ("axes" in topomap_args):
        if not (("axes" in ts_args) and ("axes" in topomap_args)):
            raise ValueError(
                "If one of `ts_args` and `topomap_args` contains "
                "'axes', the other must, too."
            )
        _validate_if_list_of_axes([ts_args["axes"]], 1)

        if times in (None, "peaks"):
            n_topomaps = 3 + 1
        else:
            assert not isinstance(times, str)
            n_topomaps = len(times) + 1

        _validate_if_list_of_axes(list(topomap_args["axes"]), n_topomaps)
        got_axes = True

    # channel selection
    # simply create a new evoked object with the desired channel selection
    # Need to deal with proj before picking to avoid bad projections
    proj = topomap_args.get("proj", True)
    proj_ts = ts_args.get("proj", True)
    if proj_ts != proj:
        raise ValueError(
            f'topomap_args["proj"] (default True, got {proj}) must match '
            f'ts_args["proj"] (default True, got {proj_ts})'
        )
    _check_option('topomap_args["proj"]', proj, (True, False, "reconstruct"))
    evoked = evoked.copy()
    if proj:
        evoked.apply_proj()
        if proj == "reconstruct":
            evoked._reconstruct_proj()
    topomap_args["proj"] = ts_args["proj"] = False  # don't reapply
    evoked.pick(picks, exclude=exclude)
    info = evoked.info
    ch_types = info.get_channel_types(unique=True, only_data_chs=True)

    # if multiple sensor types: one plot per channel type, recursive call
    if len(ch_types) > 1:
        if got_axes:
            raise NotImplementedError(
                "Currently, passing axes manually (via `ts_args` or "
                "`topomap_args`) is not supported for multiple channel types."
            )
        figs = list()
        for this_type in ch_types:  # pick only the corresponding channel type
            ev_ = evoked.copy().pick(
                [
                    info["ch_names"][idx]
                    for idx in range(info["nchan"])
                    if channel_type(info, idx) == this_type
                ]
            )
            if len(ev_.info.get_channel_types(unique=True)) > 1:
                raise RuntimeError(
                    "Possibly infinite loop due to channel "
                    "selection problem. This should never "
                    "happen! Please check your channel types."
                )
            figs.append(
                plot_evoked_joint(
                    ev_,
                    times=times,
                    title=title,
                    show=show,
                    ts_args=ts_args,
                    exclude=list(),
                    topomap_args=topomap_args,
                    cmap=cmap,
                    # linewidth=linewidth,
                )
            )
        return figs

    # set up time points to show topomaps for
    times_sec = _process_times(evoked, times, few=True)
    del times
    _, times_ts = _check_time_unit(ts_args["time_unit"], times_sec)

    # prepare axes for topomap
    if not got_axes:
        fig, ts_ax, map_ax = _prepare_joint_axes(len(times_sec), figsize=(8.0, 4.2))
        cbar_ax = None
    else:
        ts_ax = ts_args["axes"]
        del ts_args["axes"]
        map_ax = topomap_args["axes"][:-1]
        cbar_ax = topomap_args["axes"][-1]
        del topomap_args["axes"]
        fig = cbar_ax.figure

    # butterfly/time series plot
    # most of this code is about passing defaults on demand
    ts_args_def = dict(
        picks=None,
        unit=True,
        ylim=None,
        xlim="tight",
        proj=False,
        hline=None,
        units=None,
        scalings=None,
        titles=None,
        gfp=False,
        window_title=None,
        spatial_colors=True,
        zorder="std",
        sphere=None,
        draw=False,
    )
    ts_args_def.update(ts_args)
    print(ts_args_def)
    _plot_evoked(
        evoked, axes=ts_ax, cmap=cmap, 
        # linewidth=linewidth, 
        show=False, plot_type="butterfly", exclude=[], **ts_args_def
    )

    # handle title
    # we use a new axis for the title to handle scaling of plots
    old_title = ts_ax.get_title()
    ts_ax.set_title("")

    if title is not None:
        if title == "":
            title = old_title
        fig.suptitle(title)

    # topomap
    contours = topomap_args.get("contours", 6)
    ch_type = ch_types.pop()  # set should only contain one element
    # Since the data has all the ch_types, we get the limits from the plot.
    vmin, vmax = ts_ax.get_ylim()
    norm = ch_type == "grad"
    vmin = 0 if norm else vmin
    vmin, vmax = _setup_vmin_vmax(evoked.data, vmin, vmax, norm)
    if not isinstance(contours, (list, np.ndarray)):
        locator, contours = _set_contour_locator(vmin, vmax, contours)
    else:
        locator = None

    topomap_args_pass = dict(extrapolate="local") if ch_type == "seeg" else dict()
    topomap_args_pass.update(topomap_args)
    topomap_args_pass["outlines"] = topomap_args.get("outlines", "head")
    topomap_args_pass["contours"] = contours
    evoked.plot_topomap(
        times=times_sec, axes=map_ax, show=False, colorbar=False, **topomap_args_pass
    )

    if topomap_args.get("colorbar", True):
        from matplotlib import ticker

        cbar = fig.colorbar(map_ax[0].images[0], ax=map_ax, cax=cbar_ax, shrink=0.8)
        cbar.ax.grid(False)  # auto-removal deprecated as of 2021/10/05
        if isinstance(contours, (list, np.ndarray)):
            cbar.set_ticks(contours)
        else:
            if locator is None:
                locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = locator
        cbar.update_ticks()

    # connection lines
    # draw the connection lines between time series and topoplots
    for timepoint, map_ax_ in zip(times_ts, map_ax):
        con = ConnectionPatch(
            xyA=[timepoint, ts_ax.get_ylim()[1]],
            xyB=[0.5, 0],
            coordsA="data",
            coordsB="axes fraction",
            axesA=ts_ax,
            axesB=map_ax_,
            color="grey",
            linestyle="-",
            linewidth=1.5,
            alpha=0.66,
            zorder=1,
            clip_on=False,
        )
        fig.add_artist(con)

    # mark times in time series plot
    for timepoint in times_ts:
        ts_ax.axvline(
            timepoint, color="grey", linestyle="-", linewidth=1.5, alpha=0.66, zorder=0
        )

    # show and return it
    plt_show(show)
    return fig

def _plot_evoked(
    evoked,
    picks=None,
    exclude="bads",
    unit=True,
    show=True,
    ylim=None,
    proj=False,
    xlim="tight",
    hline=None,
    units=None,
    scalings=None,
    titles=None,
    axes=None,
    plot_type="butterfly",
    cmap=None,
    # linewidth=1,
    gfp=False,
    window_title=None,
    spatial_colors=False,
    selectable=True,
    zorder="unsorted",
    noise_cov=None,
    colorbar=True,
    mask=None,
    mask_style=None,
    mask_cmap=None,
    mask_alpha=0.25,
    time_unit="s",
    show_names=False,
    group_by=None,
    sphere=None,
    *,
    highlight=None,
    draw=True,
):
    """Aux function for plot_evoked and plot_evoked_image (cf. docstrings).

    Extra params are:

    plot_type : str, value ('butterfly' | 'image')
        The type of graph to plot: 'butterfly' plots each channel as a line
        (x axis: time, y axis: amplitude). 'image' plots a 2D image where
        color depicts the amplitude of each channel at a given time point
        (x axis: time, y axis: channel). In 'image' mode, the plot is not
        interactive.
    draw : bool
        If True, draw at the end.
    """
    import matplotlib.pyplot as plt

    _check_option("spatial_colors", spatial_colors, [True, False, "auto"])
    # For evoked.plot_image ...
    # First input checks for group_by and axes if any of them is not None.
    # Either both must be dicts, or neither.
    # If the former, the two dicts provide picks and axes to plot them to.
    # Then, we call this function recursively for each entry in `group_by`.
    if plot_type == "image" and isinstance(group_by, dict):
        if axes is None:
            axes = dict()
            for sel in group_by:
                plt.figure(layout="constrained")
                axes[sel] = plt.axes()
        if not isinstance(axes, dict):
            raise ValueError(
                "If `group_by` is a dict, `axes` must be " "a dict of axes or None."
            )
        _validate_if_list_of_axes(list(axes.values()))
        remove_xlabels = any([_is_last_row(ax) for ax in axes.values()])
        for sel in group_by:  # ... we loop over selections
            if sel not in axes:
                raise ValueError(
                    sel + " present in `group_by`, but not " "found in `axes`"
                )
            ax = axes[sel]
            # the unwieldy dict comp below defaults the title to the sel
            title = (
                {channel_type(evoked.info, idx): sel for idx in group_by[sel]}
                if titles is None
                else titles
            )
            _plot_evoked(
                evoked,
                group_by[sel],
                exclude,
                unit,
                show,
                ylim,
                proj,
                xlim,
                hline,
                units,
                scalings,
                title,
                ax,
                plot_type,
                cmap=cmap,
                # linewidth=linewidth,
                gfp=gfp,
                window_title=window_title,
                selectable=selectable,
                noise_cov=noise_cov,
                colorbar=colorbar,
                mask=mask,
                mask_style=mask_style,
                mask_cmap=mask_cmap,
                mask_alpha=mask_alpha,
                time_unit=time_unit,
                show_names=show_names,
                sphere=sphere,
                draw=False,
                spatial_colors=spatial_colors,
            )
            if remove_xlabels and not _is_last_row(ax):
                ax.set_xticklabels([])
                ax.set_xlabel("")
        ims = [ax.images[0] for ax in axes.values()]
        clims = np.array([im.get_clim() for im in ims])
        min_, max_ = clims.min(), clims.max()
        for im in ims:
            im.set_clim(min_, max_)
        figs = [ax.get_figure() for ax in axes.values()]
        if len(set(figs)) == 1:
            return figs[0]
        else:
            return figs
    elif isinstance(axes, dict):
        raise ValueError(
            "If `group_by` is not a dict, " "`axes` must not be a dict either."
        )

    time_unit, times = _check_time_unit(time_unit, evoked.times)
    evoked = evoked.copy()  # we modify info
    info = evoked.info
    if axes is not None and proj == "interactive":
        raise RuntimeError(
            "Currently only single axis figures are supported"
            " for interactive SSP selection."
        )

    _check_option("gfp", gfp, [True, False, "only"])

    if highlight is not None:
        highlight = np.array(highlight, dtype=float)
        highlight = np.atleast_2d(highlight)
        if highlight.shape[1] != 2:
            raise ValueError(
                f'"highlight" must be reshapable into a 2D array with shape '
                f"(n, 2). Got {highlight.shape}."
            )

    scalings = _handle_default("scalings", scalings)
    titles = _handle_default("titles", titles)
    units = _handle_default("units", units)

    if plot_type == "image":
        if ylim is not None and not isinstance(ylim, dict):
            # The user called Evoked.plot_image() or plot_evoked_image(), the
            # clim parameters of those functions end up to be the ylim here.
            raise ValueError(
                "`clim` must be a dict. " "E.g. clim = dict(eeg=[-20, 20])"
            )

    picks = _picks_to_idx(info, picks, none="all", exclude=())
    if len(picks) != len(set(picks)):
        raise ValueError("`picks` are not unique. Please remove duplicates.")

    bad_ch_idx = [
        info["ch_names"].index(ch) for ch in info["bads"] if ch in info["ch_names"]
    ]
    if len(exclude) > 0:
        if isinstance(exclude, str) and exclude == "bads":
            exclude = bad_ch_idx
        elif isinstance(exclude, list) and all(isinstance(ch, str) for ch in exclude):
            exclude = [info["ch_names"].index(ch) for ch in exclude]
        else:
            raise ValueError('exclude has to be a list of channel names or "bads"')

        picks = np.array([pick for pick in picks if pick not in exclude])

    types = np.array(info.get_channel_types(picks), str)
    ch_types_used = list()
    for this_type in _VALID_CHANNEL_TYPES:
        if this_type in types:
            ch_types_used.append(this_type)

    fig = None
    if axes is None:
        fig, axes = plt.subplots(len(ch_types_used), 1, layout="constrained")
        if isinstance(axes, plt.Axes):
            axes = [axes]
        fig.set_size_inches(6.4, 2 + len(axes))

    if isinstance(axes, plt.Axes):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = list(axes)

    if fig is None:
        fig = axes[0].get_figure()

    if window_title is not None:
        _set_window_title(fig, window_title)

    if len(axes) != len(ch_types_used):
        raise ValueError(
            "Number of axes (%g) must match number of channel "
            "types (%d: %s)" % (len(axes), len(ch_types_used), sorted(ch_types_used))
        )
    _check_option("proj", proj, (True, False, "interactive", "reconstruct"))
    noise_cov = _check_cov(noise_cov, info)
    if proj == "reconstruct" and noise_cov is not None:
        raise ValueError('Cannot use proj="reconstruct" when noise_cov is not ' "None")
    projector, whitened_ch_names = _setup_plot_projector(
        info, noise_cov, proj=proj is True, nave=evoked.nave
    )
    if len(whitened_ch_names) > 0:
        unit = False
    if projector is not None:
        evoked.data[:] = np.dot(projector, evoked.data)
    if proj == "reconstruct":
        evoked = evoked._reconstruct_proj()

    if plot_type == "butterfly":
        _plot_lines(
            evoked.data,
            info,
            picks,
            fig,
            axes,
            spatial_colors,
            unit,
            units,
            scalings,
            hline,
            gfp,
            types,
            zorder,
            xlim,
            ylim,
            times,
            bad_ch_idx,
            titles,
            ch_types_used,
            selectable,
            False,
            cmap=cmap,
            line_alpha=1.0,
            nave=evoked.nave,
            time_unit=time_unit,
            sphere=sphere,
            highlight=highlight,
        )
        plt.setp(axes, xlabel=f"Time ({time_unit})")

    elif plot_type == "image":
        for ai, (ax, this_type) in enumerate(zip(axes, ch_types_used)):
            use_nave = evoked.nave if ai == 0 else None
            this_picks = list(picks[types == this_type])
            _plot_image(
                evoked.data,
                ax,
                this_type,
                this_picks,
                unit,
                units,
                scalings,
                times,
                xlim,
                ylim,
                titles,
                colorbar=colorbar,
                cmap=cmap,
                mask=mask,
                mask_style=mask_style,
                mask_cmap=mask_cmap,
                mask_alpha=mask_alpha,
                nave=use_nave,
                time_unit=time_unit,
                show_names=show_names,
                ch_names=evoked.ch_names,
            )
    if proj == "interactive":
        _check_delayed_ssp(evoked)
        params = dict(
            evoked=evoked,
            fig=fig,
            projs=info["projs"],
            axes=axes,
            types=types,
            units=units,
            scalings=scalings,
            unit=unit,
            ch_types_used=ch_types_used,
            picks=picks,
            plot_update_proj_callback=_plot_update_evoked,
            plot_type=plot_type,
        )
        _draw_proj_checkbox(None, params)

    plt.setp(fig.axes[: len(ch_types_used) - 1], xlabel="")
    if draw:
        fig.canvas.draw()  # for axes plots update axes.
    plt_show(show)
    return fig

def _plot_lines(
    data,
    info,
    picks,
    fig,
    axes,
    spatial_colors,
    unit,
    units,
    scalings,
    hline,
    gfp,
    types,
    zorder,
    xlim,
    ylim,
    times,
    bad_ch_idx,
    titles,
    ch_types_used,
    selectable,
    psd,
    line_alpha,
    nave,
    time_unit,
    sphere,
    *,
    highlight,
    cmap=None,
):
    """Plot data as butterfly plot."""
    from matplotlib import patheffects
    from matplotlib import pyplot as plt
    from matplotlib.widgets import SpanSelector

    assert len(axes) == len(ch_types_used)
    texts = list()
    idxs = list()
    lines = list()
    sphere = _check_sphere(sphere, info)
    path_effects = [patheffects.withStroke(linewidth=2, foreground="w", alpha=0.75)]
    gfp_path_effects = [patheffects.withStroke(linewidth=5, foreground="w", alpha=0.75)]
    if selectable:
        selectables = np.ones(len(ch_types_used), dtype=bool)
        for type_idx, this_type in enumerate(ch_types_used):
            idx = picks[types == this_type]
            if len(idx) < 2 or (this_type == "grad" and len(idx) < 4):
                # prevent unnecessary warnings for e.g. EOG
                if this_type in _DATA_CH_TYPES_SPLIT:
                    logger.info(
                        "Need more than one channel to make "
                        f"topography for {this_type}. Disabling interactivity."
                    )
                selectables[type_idx] = False

    if selectable:
        # Parameters for butterfly interactive plots
        params = dict(
            axes=axes,
            texts=texts,
            lines=lines,
            ch_names=info["ch_names"],
            idxs=idxs,
            need_draw=False,
            path_effects=path_effects,
        )
        fig.canvas.mpl_connect("pick_event", partial(_butterfly_onpick, params=params))
        fig.canvas.mpl_connect(
            "button_press_event", partial(_butterfly_on_button_press, params=params)
        )
    for ai, (ax, this_type) in enumerate(zip(axes, ch_types_used)):
        line_list = list()  # 'line_list' contains the lines for this axes
        if unit is False:
            this_scaling = 1.0
            ch_unit = "NA"  # no unit
        else:
            this_scaling = 1.0 if scalings is None else scalings[this_type]
            ch_unit = units[this_type]
        idx = list(picks[types == this_type])
        idxs.append(idx)

        if len(idx) > 0:
            # Set amplitude scaling
            D = this_scaling * data[idx, :]
            _check_if_nan(D)
            gfp_only = gfp == "only"
            if not gfp_only:
                chs = [info["chs"][i] for i in idx]
                locs3d = np.array([ch["loc"][:3] for ch in chs])
                # _plot_psd can pass spatial_colors=color (e.g., "black") so
                # we need to use "is True" here
                _spat_col = _check_spatial_colors(info, idx, spatial_colors)
                if _spat_col is True and not _check_ch_locs(info=info, picks=idx):
                    warn(
                        "Channel locations not available. Disabling spatial " "colors."
                    )
                    _spat_col = selectable = False
                if _spat_col is True and len(idx) != 1:
                    x, y, z = locs3d.T
                    colors = _rgb(x, y, z)
                    _handle_spatial_colors(
                        colors, info, idx, this_type, psd, ax, sphere
                    )
                    bad_color = (0.5, 0.5, 0.5)
                else:

                    if cmap is not None:
                        if isinstance(cmap, str):
                            cmap = plt.get_cmap(cmap)
                            print(f"Using colormap {cmap}")
                        col = [cmap(i) for i in np.linspace(0, 1, len(idx))]
                    elif isinstance(_spat_col, (tuple, str)):
                        col = [_spat_col]
                    else:
                        col = ["k"]
                    bad_color = "r"
                    colors = col * len(idx)
                for i in bad_ch_idx:
                    if i in idx:
                        colors[idx.index(i)] = bad_color

                if zorder == "std":
                    # find the channels with the least activity
                    # to map them in front of the more active ones
                    z_ord = D.std(axis=1).argsort()
                elif zorder == "unsorted":
                    z_ord = list(range(D.shape[0]))
                elif not callable(zorder):
                    error = (
                        '`zorder` must be a function, "std" ' 'or "unsorted", not {0}.'
                    )
                    raise TypeError(error.format(type(zorder)))
                else:
                    z_ord = zorder(D)

                # plot channels
                for ch_idx, z in enumerate(z_ord):
                    line_list.append(
                        ax.plot(
                            times,
                            D[ch_idx],
                            picker=True,
                            zorder=z + 1 if _spat_col else 1,
                            color=colors[ch_idx],
                            alpha=line_alpha,
                            linewidth=1.5,
                        )[0]
                    )
                    line_list[-1].set_pickradius(3.0)
                line_legend=True # hard code for now
                if line_legend:
                    # use default matplotlib legend
                    plt.legend(
                        [line_list[i] for i in range(len(line_list))],
                        [info["ch_names"][idx[i]] for i in range(len(line_list))],
                        fontsize=8,
                        frameon=True,
                        framealpha=1,
                        loc='center right'
                    )                 
                # add horizontal line at y=0, and dashed vertial line at x=0
                ax.axhline(0, linestyle="--", linewidth=0.5, color="k")
                ax.axvline(0, linestyle="--", linewidth=0.5, color="k")

            # Plot GFP / RMS
            if gfp:
                if gfp in [True, "only"]:
                    if this_type == "eeg":
                        this_gfp = D.std(axis=0, ddof=0)
                        label = "GFP"
                    else:
                        this_gfp = np.linalg.norm(D, axis=0) / np.sqrt(len(D))
                        label = "RMS"

                gfp_color = 3 * (0.0,) if spatial_colors is True else (0.0, 1.0, 0.0)
                this_ylim = (
                    ax.get_ylim()
                    if (ylim is None or this_type not in ylim.keys())
                    else ylim[this_type]
                )
                if gfp_only:
                    y_offset = 0.0
                else:
                    y_offset = this_ylim[0]
                this_gfp += y_offset
                ax.autoscale(False)
                ax.fill_between(
                    times,
                    y_offset,
                    this_gfp,
                    color="none",
                    facecolor=gfp_color,
                    zorder=1,
                    alpha=0.2,
                )
                line_list.append(
                    ax.plot(
                        times, this_gfp, color=gfp_color, zorder=3, alpha=line_alpha
                    )[0]
                )
                ax.text(
                    times[0] + 0.01 * (times[-1] - times[0]),
                    this_gfp[0] + 0.05 * np.diff(ax.get_ylim())[0],
                    label,
                    zorder=4,
                    color=gfp_color,
                    path_effects=gfp_path_effects,
                )
            for ii, line in zip(idx, line_list):
                if ii in bad_ch_idx:
                    line.set_zorder(2)
                    if spatial_colors is True:
                        line.set_linestyle("--")
            ax.set_ylabel(ch_unit)
            texts.append(
                ax.text(
                    0,
                    0,
                    "",
                    zorder=3,
                    verticalalignment="baseline",
                    horizontalalignment="left",
                    fontweight="bold",
                    alpha=0,
                    clip_on=True,
                )
            )

            if xlim is not None:
                if xlim == "tight":
                    xlim = (times[0], times[-1])
                ax.set_xlim(xlim)
            if ylim is not None and this_type in ylim:
                ax.set_ylim(ylim[this_type])
            ax.set(
                title=r"%s (%d channel%s)" % (titles[this_type], len(D), _pl(len(D)))
            )
            if ai == 0:
                _add_nave(ax, nave)
            if hline is not None:
                for h in hline:
                    c = "grey" if spatial_colors is True else "r"
                    ax.axhline(h, linestyle="--", linewidth=2, color=c)

            # Plot highlights
            if highlight is not None:
                this_ylim = (
                    ax.get_ylim()
                    if (ylim is None or this_type not in ylim.keys())
                    else ylim[this_type]
                )
                for this_highlight in highlight:
                    ax.fill_betweenx(
                        this_ylim,
                        this_highlight[0],
                        this_highlight[1],
                        facecolor="orange",
                        alpha=0.15,
                        zorder=99,
                    )
                # Put back the y limits as fill_betweenx messes them up
                ax.set_ylim(this_ylim)

        lines.append(line_list)

    # if selectable:
    #     for ax in np.array(axes)[selectables]:
    #         if len(ax.lines) == 1:
    #             continue
    #         text = ax.annotate(
    #             "Loading...",
    #             xy=(0.01, 0.1),
    #             xycoords="axes fraction",
    #             fontsize=20,
    #             color="green",
    #             zorder=3,
    #         )
    #         text.set_visible(False)
    #         callback_onselect = partial(
    #             _line_plot_onselect,
    #             ch_types=ch_types_used,
    #             info=info,
    #             data=data,
    #             times=times,
    #             text=text,
    #             psd=psd,
    #             time_unit=time_unit,
    #             sphere=sphere,
    #         )
    #         blit = False if plt.get_backend() == "MacOSX" else True
    #         minspan = 0 if len(times) < 2 else times[1] - times[0]
    #         rect_kw = _prop_kw("rect", dict(alpha=0.5, facecolor="red"))
    #         ax._span_selector = SpanSelector(
    #             ax,
    #             callback_onselect,
    #             "horizontal",
    #             minspan=minspan,
    #             useblit=blit,
    #             **rect_kw,
    #         )

