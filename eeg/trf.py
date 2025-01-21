""" 
# This module is based on mne linear regression code.
# It is from the code accompanying the paper https://doi.org/10.1073/pnas.2201968119 
# M. Heilbron, K. Armeni, J. Schoffelen, P. Hagoort, F.P. de Lange, A hierarchy of linguistic predictions during natural language comprehension, Proc. Natl. Acad. Sci. U.S.A.
# 119 (32) e2201968119,
# https://doi.org/10.1073/pnas.2201968119 (2022).
# orig docstring:
all routines in this module are personal functions, 
many are hacks to / adjustments or / built upon mne/stats/regression.py
linear_regression_raw
"""



import numpy as np
import pandas as pd

import copy
import itertools
import operator

from scipy import sparse
from typing import Iterable,Sequence

from mne.io.pick import pick_types, pick_info
from mne.utils import _reject_data_segments

from mne.evoked import Evoked, EvokedArray
from mne.io import RawArray
from mne.decoding import TimeDelayingRidge

from sklearn.linear_model import Ridge,RidgeCV
from sklearn.model_selection import cross_val_score

from IPython.core.debugger import set_trace

# import eelbrain 
# from eelbrain import NDVar,Case


#################################################################################################
def estimate_trf_laplacian(raw_in,covs_in:pd.DataFrame,onset_times,tmin,tmax,
                         reg_type='laplacian',reg_alpha=1):
    """estiamte trf via TimeDelayedRidge.
    
    todo: tuning?
    todo: docs 
    todo: multiple intercepts (first check how much it matters :))
    """
    
    X,y,cov_names=_get_Xy(copy.deepcopy(covs_in),onset_times,raw_in)
    
    tdl=TimeDelayingRidge(tmin,tmax,raw_in.info['sfreq'],alpha=reg_alpha,
                  reg_type=reg_type,fit_intercept=False)
    tdl.fit(X,y)
    
    # get inputs 
    evk_ins=_get_evk_inputs(covs_in,cov_names,tmin,tmax,copy.deepcopy(raw_in.info))

    # return the evoked dict 
    return(_make_evoked_dict(np.copy(tdl.coef_),**evk_ins))


def estimate_trf_boosting(raw_in,covs_in:pd.DataFrame,onset_times,tmin,tmax,
                         basis_dur=.25,n_parts=4,delta=0.05,error='l2',scale_data=False):
    """estimate TRF by boosting. dependencies: eelbrain, and NDVar (from eelbrain imported)
    
    WARNING: coordinate descent -- SLOW!!!!!!!!! unusuably slow for large chunks of data (hours)
    
    tmin:float
    tmax:float
    basis_dur:width of hamming basis func (in sec)
    n_parts=:number of partitions 
    delta:step_size
    error:str: ('l1' or 'l2') -- default: l2
    error:bool or 'inplace' -- default:False (assumes data is already zscored)
    
    output:
    -res: BoostingResult 
        result instance from eelbrain 
    -evoked_dict : Dict of EvokedArrays:
        easily averagable EEG representation of same results 
    
    #####################################################################
    -- consider returning dict with XY for easy cross-val at later stage, i.e. fitting only once.
    #####################################################################
    """
    # convert RawArray to NDVar
    raw_ndvar=_raw2ndvar(raw_in)
    
    # get (not time-expanded X and y matrix)
    X,y,cov_names=_get_Xy(copy.deepcopy(covs_in),onset_times,raw_in)

    # wrap covariates into dict of NDVars 
    nd_vars={cov_key:NDVar(X[:,cov_i].reshape(-1),dims=(raw_ndvar.time,),name=cov_key) 
            for cov_i,cov_key in enumerate(cov_names)}

    # perform boosting and get results 
    res=eelbrain.boosting(raw_ndvar,[nd_vars[k] for k in cov_names],tmin,tmax,error=error,
                          scale_data=scale_data,basis=basis_dur,partitions=n_parts,delta=delta)

    # get ingredients to construct EvokedArray
    evk_ins=_get_evk_inputs(covs_in,cov_names,tmin,tmax,copy.deepcopy(raw_in.info))
    evk_dict=_make_evoked_dict(_boost_res2coefs_ar(res),**evk_ins)

    return(res,evk_dict)


##### helpers for boosting and laplace variety

def _raw2ndvar(raw_ar):
    """eelbrain ndvar from mne raw array"""
    t_min=raw_ar.times[0]
    t_ntps=len(raw_ar.times)
    t_step=np.median(np.diff(raw_ar.times))

    time_uts=eelbrain.UTS(t_min,t_step,t_ntps)
    return(eelbrain.NDVar(raw_ar._data,dims=(eelbrain.Case,time_uts)))

def _boost_res2coefs_ar(b_res) -> np.ndarray:
    """from tuple of boosting results, construct array to make evoked"""
    # construct empty array, shape (n_covs,n_chans,n_delays)
    coefs_ar=np.zeros((len(b_res.h),b_res.h[0].x.shape[0],b_res.h[0].x.shape[1]))
    for h_i,h in enumerate(b_res.h):
        coefs_ar[h_i,:,:]=b_res.h[h_i].x # deep copy 
    return(coefs_ar)


def _get_Xy(all_covs:dict,all_onsets:dict,raw_arr:RawArray):
    """
    from given covariates, onsets and RawArray, get X and y matrix 
    (NOTE: ***NOT the time-expanded version***).
    
    if no 'onset' (intercept) column in covs,then automatically added.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_Xy_single(covs_df:pd.DataFrame,onset_times:np.ndarray,raw_arr:RawArray,
                   prefix=''):
        """helper functions of _get_Xy"""
        # add intercept called '{prefix_}onset' 
        covs_df.insert(0,prefix+'onset',np.ones_like(np.array(covs_df.iloc[:,0]),dtype=int))
        for x_i,(x_lbl,x_dat) in enumerate(covs_df.iteritems()):
            this_col=np.zeros_like(raw_arr.times)
            this_col[raw_arr.time_as_index(onset_times,use_rounding=True)]=x_dat.to_numpy()
            X=this_col.reshape(-1,1) if x_i<1 else np.hstack((X,this_col.reshape(-1,1)))

        # out X,y,cov_names 
        # shape: (n_samps,n_feats),(n_samps,n_outputs),[n_feats]
        return(X,raw_arr._data.T,covs_df.columns.to_list())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    for ev_i,(this_cov,these_onsets) in enumerate(zip(all_covs.values(),all_onsets.values())):
        this_pref=this_cov.columns[ev_i].split('_')[0]+'_' if len(all_covs)>1 else ''
        this_X,y,these_names=_get_Xy_single(this_cov,these_onsets,raw_arr,this_pref)
        if ev_i==0:
            unified_X=np.copy(this_X)
            unified_names=these_names
        else:
            unified_X=np.hstack((unified_X,this_X))
            unified_names+=these_names
    
    return(unified_X,y,unified_names)


def _get_evk_inputs(cov_df,cov_names,tmin,tmax,info_dct):
    """output has to be dict with keys :)"""
    # assuming all thesame intercept length
    evk_ins=dict(
        cov_names =cov_names,
        cov_length={cov_k:len(cov_df) for cov_k in cov_names},
        tmin_samp ={cov_key:round(tmin*info_dct['sfreq']) for cov_key in cov_names},
        tmax_samp ={cov_key:round(tmax*info_dct['sfreq'])+1 for cov_key in cov_names},
        info=info_dct) 
    return(evk_ins)
    

def _make_evoked_dict(coefs_in, cov_names, cov_length, tmin_samp, tmax_samp, info):
    """Create a dictionary of Evoked objects.
    These will be created from a coefs matrix and condition durations.
    
    (for ***TimeDelayingRidgeImplementation***)
    
    ~~~~~~~~~~~~~
    coefs_in: nparray, shape (n_coefs,n_chans,Nn_delays)
        deepcopy from e.g. TimeDelayedRidge.coef_
    conds: list of cond_names
        e.g. ['onset','lexical_surprise']
    cov_length: dict: {'cov':len(cov)}, 
        e.g. {'onset': 576, 'lexical_surprise': 576}
    tmin_samp: dict: tmin_sample
        for instance: {'onset': -13, 'lexical_surprise': -13}
    tmax_s: dict: tmin_sampample 
        for instacne: {'onset': 129, 'lexical_surprise': 129}
    info: raw_info array
        
    ~~~~~~~~~~~~~
    """
    evokeds = dict()
    # in case output was from TimeDelayedRidge, move the axes 
    coefs=coefs_in if coefs_in.shape[1]>coefs_in.shape[0] else np.moveaxis(coefs_in,1,0)
        
    for c_i,cov_lbl in enumerate(cov_names):
        tmin_, tmax_ = tmin_samp[cov_lbl], tmax_samp[cov_lbl]
        evokeds[cov_lbl] = EvokedArray(
            coefs[c_i,:,:], info=info, comment=cov_lbl,
            tmin=tmin_ / float(info["sfreq"]), nave=cov_length[cov_lbl],
            kind='average')  # nave and kind are not really correct but useful placeholders 
    return evokeds
#######################################################################################################

## TODO: make cross-val-score routine

def estimate_trf(raw, events_in, event_id_in=None,covariates_in=None,
                tmin=-.1, tmax=1,reject=None, flat=None, tstep=1.,
                decim=1, picks=None, ridge=Ridge(alpha=1,fit_intercept=False)):
    """adaptation of MNE TRF function that handles multiple events in one parametric model.
    
    Note: function can be used for models with one intercept, but not for models with multiple
    binary (categorical) predictors without covariates. 
    
    returns mne.Evoked dict for every TRF.
    
    input:
    
    Parameters (unique)
    ----------
    events_in   : array-like or Dict
        mne events array (if one intercept) OR dict with multiple event_arrays 
    event_id_in : dict or nested-dict 
        dict with {event_name : event_code} (if one intercept) OR 
        (nested dict, eg: event_id_in[event_type]={event_name : event_code}.
    covarites_in: PD dataframe or Dict of pd dataframe
        dataframe with covariates (if one intercept) OR
        dict of dataframe with covariates for each intercept. 
    ---------
    other parameters (from MNE version)
    
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
        or cleaning raw data before the regression is performed: set up
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
    ridge : callable
        Function which takes as its inputs the sparse predictor
        matrix X and the observation matrix Y, and returns the coefficient
        matrix b. default: ridge with alpha=1.
    Returns
    -------
    evokeds : dict
        A dict where the keys correspond to conditions and the values are
        Evoked objects with the ER[F/P]s. These can be used exactly like any
        other Evoked object, including e.g. plotting or statistics.
    """
    # prepare regression model and clean data 
    X, data, evk_info=_get_trf_Xy(raw,events_in,#return_evoked_info=True,
                                         event_id_in=event_id_in,covariates_in=covariates_in,
                                         tmin=tmin, tmax=tmax,reject=reject, flat=flat,
                                         tstep=tstep,decim=decim, picks=picks)
    
    # solve linear system
    coefs = ridge.fit(X, data.T).coef_

    if coefs.shape[0] != data.shape[0]:
        raise ValueError("solver output has unexcepted shape. Supply a "
                         "function that returns coefficients in the form "
                         "(n_targets, n_features), where targets == channels.")

    # construct Evoked objects to be returned from output
    evokeds = _make_evokeds(coefs, evk_info['conds'], evk_info['cond_length'], 
                            evk_info['tmin_s'], evk_info['tmax_s'], evk_info['info'])
    
    return(evokeds)

# helper function 
def _get_trf_Xy(raw, events_in,clean_inputs=True,
                event_id_in=None,covariates_in=None,tmin=-.1, tmax=1,
                reject=None, flat=None, tstep=1.,decim=1, picks=None,permute=None):
    """prepare design matrix (X), data matrix (y) and other inputs for TRF estimation or CV-scoring.
    
    Note, this function can handle multiple events of unequal lengths and multiple covariates "around" each events.
    
        
    Note: if asterisk is added to event_name, we don't add time-expanded intercept
    
    ------------        
    Parameters (specific to adaptation)
    ----------
    - raw: RawArray object
        raw data (data shall contain no discontinuities)
    - events_in   : array-like or Dict {event_name: onsets, event_name: onsets}
        mne events array (if one intercept) OR dict with multiple event_arrays 
    - covarites_in: None OR DataFrame OR Dict of DataFrames: {event_name: pd.DataFrame,...}
        None if no covariates 
        dataframe with covariates (if one intercept) OR
        dict of dataframe with covariates for each intercept. 
    - clean_inputs: Bool (default: True)
        remove periods with no events defined or nans (after constructing full matrix)
    - return_evoked_info: bool
        controls whether or not to return additional info necessary to create mne.Evoked object. 
        If False, only X and y are returned.  
        
        
    Returns
   -----------------
   X,y,[evoked_info] (optional)


    ---------
    other parameters (from MNE version)
    
    event_id_in : dict or nested-dict 
        dict with {event_name : event_code} (if one intercept) OR 
        (nested dict, eg: event_id_in[event_type]={event_name : event_code}.
        (If you want specific codes for some reason).
        
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
        or cleaning raw data before the regression is performed: set up
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
    ridge : callable
        Function which takes as its inputs the sparse predictor
        matrix X and the observation matrix Y, and returns the coefficient
        matrix b. default: ridge with alpha=1.
    """
    # HANDLE VARIABLE INPUT FORMS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # below some preamble that deals with input having either a single event
    # (i.e. 1 time-expanded intercept) or multiple events (multiple intercepts). 

    # in case event_id was omitted, 
    if event_id_in is None: # try to derive from events_in 
        event_id_in = {str(v): v for v in set(events_in[:, 2])}

    # check consistency 
    if isinstance(events_in,dict):
        consistent=((any(isinstance(i,dict) for i in event_id_in.values()))
                   and (isinstance(covariates_in,dict)))
    else:
        consistent=~(((any(isinstance(i,dict) for i in event_id_in.values()))
                   or (isinstance(covariates_in,dict))))

    assert consistent, ValueError('events, event_id and covariates (nesting) not consistent!')

    # if the input is a single event (as in MNE), wrap everything in dict so we can loop
    if not isinstance(events_in,dict):
        if len(event_id_in)>1: raise ValueError('cant handle multiple binary (non-intercept) events!')
        event_id_in={k:event_id_in for k in event_id_in}     # make event_id nested dict 
        events_in = {k:events_in for k in event_id_in}       # make events dict 
        covariates_in={k:covariates_in for k in event_id_in} # make covarites dict 

    # Create time-expanded design matrix
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # below we create a design matrix for each event-type. and then concatenate them

    # loop over types and ids
    conds=[];cond_length={};itr=0;tmin_s={};tmax_s={}
    for event_type,event_id in event_id_in.items():
        
        # if asterisk is in the model, 
        add_const=True if '*' not in event_type else False
        events,covariates=events_in[event_type],covariates_in[event_type]

        # build data (downsample if needed, select specific channel_types)
        data, info, events = _prepare_rerp_data(raw, events, picks=picks,
                                            decim=decim)

        # build predictors --- tr-ex?
        this_X, these_conds, this_cond_length, this_tmin_s, this_tmax_s = _prepare_rerp_preds(
            n_samples=data.shape[1], sfreq=info["sfreq"], events=events,
            event_id=event_id, tmin=tmin, tmax=tmax, covariates=covariates,add_const=add_const)

        # update / concatenate conditions and design matrix 
        tmin_s.update(this_tmin_s);tmax_s.update(this_tmax_s)
        cond_length.update(this_cond_length);
        conds+=these_conds;

        X=sparse.hstack((X,this_X)) if itr>0 else this_X
        itr=+1
    
    # remove "empty" and contaminated data points
    if clean_inputs:
        X, data = _clean_rerp_input(X, data, reject, flat, decim, info, tstep) 
        X, data = _scrub_nans_X(X,data) # scrub timepoints with contaminated X
    
    evoked_info=dict(conds=conds, cond_length=cond_length, 
                     tmin_s=tmin_s, tmax_s=tmax_s,info=info,
                     reject=reject,flat=flat,tstep=tstep)
    
    if isinstance(permute,Sequence):
        if isinstance(permute,str): permute=[permute]
        for this_perm in permute:
            X=_permute_var(X,this_perm,evoked_info)

    return(X,data,evoked_info) # if return_evoked_info else (X,data)


########################################
#### helper functions with only minimal (but important) changes 
########################################

# NB: some changes are made for my specific downstream needs so these routines cannot just
# be swapped e.g. instalilng a newer version of MNE> 
def _prepare_rerp_data(raw, events, picks=None, decim=1):
    """Prepare events and data, primarily for `linear_regression_raw`."""
    if picks is None:
        picks = pick_types(raw.info, meg=True, eeg=True, ref_meg=True)
    info = pick_info(raw.info, picks)
    decim = int(decim)
    info["sfreq"] /= decim
    data, times = raw[:]
    data = data[picks, ::decim]
    if len(set(events[:, 0])) < len(events[:, 0]):
        raise ValueError("`events` contains duplicate time points. Make "
                         "sure all entries in the first column of `events` "
                         "are unique.")

    events = events.copy()
    events[:, 0] -= raw.first_samp
    events[:, 0] //= decim
    if len(set(events[:, 0])) < len(events[:, 0]):
        raise ValueError("After decimating, `events` contains duplicate time "
                         "points. This means some events are too closely "
                         "spaced for the requested decimation factor. Choose "
                         "different events, drop close events, or choose a "
                         "different decimation factor.")

    return data, info, events



def _prepare_rerp_preds(n_samples, sfreq, events, event_id=None, tmin=-.1,
                        tmax=1, covariates=None,add_const=True):
    """Build predictor matrix and metadata (e.g. condition time windows).
    
    crucial edit:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if add_const is true, event is added as time-expanded intercept. 
    If false, then not.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if (add_const==False) and (covariates is None):
        raise ValueError('event neither binary nor continuous?')
        
    # perhaps say here, intc
    conds = list(event_id) if add_const else []
    if covariates is not None:
        conds += list(covariates)

    # time windows (per event type) are converted to sample points from times
    # int(round()) to be safe and match Epochs constructor behavior
    if isinstance(tmin, (float, int)):
        tmin_s = dict((cond, int(round(tmin * sfreq))) for cond in conds)
    else:
        tmin_s = dict((cond, int(round(tmin.get(cond, -.1) * sfreq)))
                      for cond in conds)
    if isinstance(tmax, (float, int)):
        tmax_s = dict(
            (cond, int(round((tmax * sfreq)) + 1)) for cond in conds)
    else:
        tmax_s = dict((cond, int(round(tmax.get(cond, 1.) * sfreq)) + 1)
                      for cond in conds)

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
            ids = ([event_id[cond]]
                   if isinstance(event_id[cond], int)
                   else event_id[cond])
            onsets = -(events[np.in1d(events[:, 2], ids), 0] + tmin_)
            values = np.ones((len(onsets), n_lags))

        else:  # for predictors from covariates, e.g. continuous ones
            covs = covariates[cond]
            if len(covs) != len(events):
                error = ("Condition {0} from ```covariates``` is "
                         "not the same length as ```events```").format(cond)
                raise ValueError(error)
            onsets = -(events[np.where(covs != 0), 0] + tmin_)[0]
            v = np.asarray(covs)[np.nonzero(covs)].astype(float)
            values = np.ones((len(onsets), n_lags)) * v[:, np.newaxis]

        cond_length[cond] = len(onsets)
        xs.append(sparse.dia_matrix((values, onsets),
                                    shape=(n_samples, n_lags)))

    return sparse.hstack(xs), conds, cond_length, tmin_s, tmax_s

def _clean_rerp_input(X, data, reject, flat, decim, info, tstep):
    """Detect timepoints with empty/contaminated DATA. Then remove those timepoints from y & X.
    (this does not detect tps w/ scrubbed PREDICTORS. this happens afterwrds. see: _scrub_nans_X 
    """
    # find only those positions where at least one predictor isn't 0
    has_val = np.unique(X.nonzero()[0])
    
    # find all nans in the data; format: [(start_i,stop_i),(start_i,stop_i)]
    inds=_get_nan_inds(data)
    
    # reject positions based on extreme steps in the data
    if reject is not None:
        _, extr_inds = _reject_data_segments(data, reject, flat, decim=None,
                                             info=info, tstep=tstep)
        inds+=extr_inds
        
    for t0, t1 in inds:
        has_val = np.setdiff1d(has_val, range(t0, t1))
            
    return X.tocsr()[has_val], data[:, has_val]

def _get_nan_inds(A):
    """returns inds of columns with nans. 
    """
    ### TODO: add any row, maybe with np.unique(X.nonzero()[0])
    nans_found=np.isnan(A[0,:]) if A.ndim>1 else np.isnan(A) 
    return(_get_nonzero_inds(nans_found))

def _get_nonzero_inds(col_in):
    """ return clusters of nonzero values embedded in long sparse vector 
    output: (start,stop), where stop=last_ind+1
    """
    non_zeros=[[i for i,value in it] for key,it in 
               itertools.groupby(enumerate(col_in), key=operator.itemgetter(1)) if key != 0]
    return([(a[0],a[-1]+1) for a in non_zeros]) 

def _scrub_nans_X(X,y):
    """find nan values (scrubbed/outlier timepoints) in X matrix. then delete from X & y. 
    to be performed after _clean_rerp_preds. 
     in:
    -X : scipy.sparse.csr.csr_matrix, shape(n_tps,n_feats)
    -y : numpy matrix. shape(n_resps,n_tps)
    out: 
    X, y"""
    nan_rows=np.unique(np.isnan(X.todense()).nonzero()[0])
    samps2keep=np.setdiff1d(range(X.shape[0]),nan_rows)
    return(X[samps2keep,:],y[:,samps2keep])


def _make_evokeds(coefs, conds, cond_length, tmin_s, tmax_s, info):
    """Create a dictionary of Evoked objects.
    These will be created from a coefs matrix and condition durations.
    """
    evokeds = dict()
    cumul = 0
    for cond in conds:
        tmin_, tmax_ = tmin_s[cond], tmax_s[cond]
        evokeds[cond] = EvokedArray(
            coefs[:, cumul:cumul + tmax_ - tmin_], info=info, comment=cond,
            tmin=tmin_ / float(info["sfreq"]), nave=cond_length[cond],
            kind='average')  # nave and kind are technically incorrect
        cumul += tmax_ - tmin_
    return evokeds

def _get_col_indices(evk_info):
    """Create a dictionary of Evoked objects.
    These will be created from a coefs matrix and condition durations.
    """
    conds, cond_length,tmin_s,tmax_s =(evk_info['conds'],evk_info['cond_length'],
                                       evk_info['tmin_s'], evk_info['tmax_s'])
    inds = dict()
    cumul = 0
    for cond in conds:
        tmin_, tmax_ = tmin_s[cond], tmax_s[cond]
        inds[cond]=(cumul,cumul + tmax_s[cond] - tmin_s[cond])
        cumul += tmax_ - tmin_
    return inds

def _permute_var(X_in,var2shuff,evk_inf):
    """shuffle variable of interest (for \delta R analyses)"""

    idxs=_get_col_indices(evk_inf)[var2shuff]
    
    X_=copy.deepcopy(X_in)
    subpart=X_[:,idxs[0]:idxs[1]]
    subpart_shuff=shuffle_nonzero_rows(subpart.T).T
    X_in[:,idxs[0]:idxs[1]]=subpart_shuff
    
    return(X_in)

def shuffle_nonzero_rows(a): # 
    """shuffle only the nonzero elements in a (sparse) matrix along rows 
    (no mixing of rows).
    useful for shuffling time-expanded design matrices "across the diagonals"
    """
    i, j = np.nonzero(a.astype(bool))
    k = np.argsort(i + np.random.rand(i.size))
    a[i,j] = a[i,j[k]]
    return a
