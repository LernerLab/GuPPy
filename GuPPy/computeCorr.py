import os
import glob
import re
import math
import h5py
import logging
import numpy as np
import pandas as pd
from scipy import signal

def insertLog(text, level):
    file = os.path.join('.','..','guppy.log')
    format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    infoLog = logging.FileHandler(file)
    infoLog.setFormatter(format)
    infoLog
    logger = logging.getLogger(file)
    logger.setLevel(level)
    
    if not logger.handlers:
        logger.addHandler(infoLog)
        if level == logging.DEBUG:
            logger.debug(text)
        if level == logging.INFO:
            logger.info(text)
        if level == logging.ERROR:
            logger.exception(text)
        if level == logging.WARNING:
            logger.warning(text)
    
    infoLog.close()
    logger.removeHandler(infoLog)

def make_dir(filepath):
	op = os.path.join(filepath, "cross_correlation_output")
	if not os.path.exists(op):
		os.mkdir(op)
	return op

# function to read hdf5 file
def read_hdf5(event, filepath, key):
    if event:
        op = os.path.join(filepath, event+'.hdf5')
    else:
        op = filepath

    if os.path.exists(op):
        with h5py.File(op, 'r') as f:
            arr = np.asarray(f[key])
    else:
        insertLog(f"{event}.hdf5 file does not exist", logging.ERROR)
        raise Exception('{}.hdf5 file does not exist'.format(event))
    
    return arr

# function to read h5 file and make a dataframe from it
def read_Df(filepath, event, name):
	if name:
		op = os.path.join(filepath, event+'_{}.h5'.format(name))
	else:
		op = os.path.join(filepath, event+'.h5')
	df = pd.read_hdf(op, key='df', mode='r')

	return df

# same function used to store PSTH in computePsth file
# Here, cross correlation dataframe is saved instead of PSTH
# cross correlation dataframe has the same structure as PSTH file
def create_Df(filepath, event, name, psth, columns=[]):
	if name:
		op = os.path.join(filepath, event+'_{}.h5'.format(name))
	else:
		op = os.path.join(filepath, event+'.h5')
	
	# check if file already exists
	#if os.path.exists(op):
	#	return 0

	# removing psth binned trials
	columns = list(np.array(columns, dtype='str'))
	regex = re.compile('bin_*')
	single_trials_index = [i for i in range(len(columns)) if not regex.match(columns[i])]
	single_trials_index = [i for i in range(len(columns)) if columns[i]!='timestamps']

	psth = psth.T
	if psth.ndim > 1:
		mean = np.nanmean(psth[:,single_trials_index], axis=1).reshape(-1,1)
		err = np.nanstd(psth[:,single_trials_index], axis=1)/math.sqrt(psth[:,single_trials_index].shape[1])
		err = err.reshape(-1,1)
		psth = np.hstack((psth,mean))
		psth = np.hstack((psth, err))
		#timestamps = np.asarray(read_Df(filepath, 'ts_psth', ''))
		#psth = np.hstack((psth, timestamps))
	try:
		ts = read_hdf5(event, filepath, 'ts')
		ts = np.append(ts, ['mean', 'err'])
	except:
		ts = None

	if len(columns)==0:
		df = pd.DataFrame(psth, index=None, columns=ts, dtype='float32')
	else:
		columns = np.asarray(columns)
		columns = np.append(columns, ['mean', 'err'])
		df = pd.DataFrame(psth, index=None, columns=columns, dtype='float32')

	df.to_hdf(op, key='df', mode='w')


def getCorrCombinations(filepath, inputParameters):
    selectForComputePsth = inputParameters['selectForComputePsth']
    if selectForComputePsth=='z_score':
        path = glob.glob(os.path.join(filepath, 'z_score_*'))
    elif selectForComputePsth=='dff':
        path = glob.glob(os.path.join(filepath, 'dff_*'))
    else:
        path = glob.glob(os.path.join(filepath, 'z_score_*')) + glob.glob(os.path.join(filepath, 'dff_*'))
    
    names = list()
    type = list()
    for i in range(len(path)):
        basename = (os.path.basename(path[i])).split('.')[0]
        names.append(basename.split('_')[-1])
        type.append((os.path.basename(path[i])).split('.')[0].split('_'+names[-1], 1)[0])
    
    names = np.unique(np.array(names))
    type = np.unique(np.array(type))

    corr_info = list()
    if len(names)<=1:
        insertLog("Cross-correlation cannot be computed because only one signal is present.", 
                    logging.INFO)
        print("Cross-correlation cannot be computed because only one signal is present.")
        return corr_info, type
    elif len(names)==2:
        corr_info = names
    else:
        corr_info = names
        corr_info.append(names[0])
    
    return corr_info, type
    


def helperCrossCorrelation(arr_A, arr_B, sample_rate):
    cross_corr = list()
    for (a, b) in zip(arr_A, arr_B):
        corr = signal.correlate(a, b)
        corr_norm = corr/ np.max(np.abs(corr))
        cross_corr.append(corr_norm)
        lag = signal.correlation_lags(len(a), len(b))
        lag_msec = np.array(lag / sample_rate, dtype='float32')
    
    cross_corr_arr = np.array(cross_corr, dtype='float32')
    lag_msec = lag_msec.reshape(1,-1)
    cross_corr_arr = np.concatenate((cross_corr_arr, lag_msec), axis=0)
    
    return cross_corr_arr


def computeCrossCorrelation(filepath, event, inputParameters):
    isCompute = inputParameters['computeCorr']
    if isCompute==True:
        corr_info, type = getCorrCombinations(filepath, inputParameters)
        if 'control' in event.lower() or 'signal' in event.lower():
            return
        else:
            for i in range(1, len(corr_info)):
                print("Computing cross-correlation for event {}...".format(event))
                insertLog(f"Computing cross-correlation for event {event}", logging.DEBUG)
                for j in range(len(type)):
                    psth_a = read_Df(filepath, event+'_'+corr_info[i-1], type[j]+'_'+corr_info[i-1])
                    psth_b = read_Df(filepath, event+'_'+corr_info[i], type[j]+'_'+corr_info[i])
                    sample_rate = 1/(psth_a['timestamps'][1]-psth_a['timestamps'][0])
                    psth_a = psth_a.drop(columns=['timestamps', 'err', 'mean'])
                    psth_b = psth_b.drop(columns=['timestamps', 'err', 'mean'])
                    cols = list(psth_a.columns)
                    arr_A, arr_B = np.array(psth_a).T, np.array(psth_b).T
                    cross_corr = helperCrossCorrelation(arr_A, arr_B, sample_rate)
                    cols.append('timestamps')
                    create_Df(make_dir(filepath), 'corr_'+event, type[j]+'_'+corr_info[i-1]+'_'+corr_info[i], cross_corr, cols)
                insertLog(f"Cross-correlation for event {event} computed.", logging.INFO)
                print("Cross-correlation for event {} computed.".format(event))
    
