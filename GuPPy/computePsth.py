# coding: utf-8

import os
import sys
import json
import glob
import re
import time
import numpy as np 
import h5py
import math
import pandas as pd
from itertools import repeat
import multiprocessing as mp
from scipy import signal as ss
from collections import OrderedDict
from preprocess import get_all_stores_for_combining_data


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
		raise Exception('{}.hdf5 file does not exist'.format(event))

	return arr

# function to write hdf5 file
def write_hdf5(data, event, filepath, key):
	op = os.path.join(filepath, event+'.hdf5')
	
	# if file does not exist create a new file
	if not os.path.exists(op):
		with h5py.File(op, 'w') as f:
			if type(data) is np.ndarray:
				f.create_dataset(key, data=data, maxshape=(None, ), chunks=True)
			else:
				f.create_dataset(key, data=data)
	# if file already exists, append data to it or add a new key to it
	else:
		with h5py.File(op, 'r+') as f:
			if key in list(f.keys()):
				if type(data) is np.ndarray:
					f[key].resize(data.shape)
					arr = f[key]
					arr[:] = data
				else:
					arr = f[key]
					arr = data
			else:
				f.create_dataset(key, data=data, maxshape=(None, ), chunks=True)


def create_Df_area_peak(filepath, arr, name, index=[]):

	op = os.path.join(filepath, 'peak_AUC_'+name+'.h5')
	dirname = os.path.dirname(filepath)

	df = pd.DataFrame(arr, index=index)

	df.to_hdf(op, key='df', mode='w')

def read_Df_area_peak(filepath, name):
	op = os.path.join(filepath, 'peak_AUC_'+name+'.h5')
	df = pd.read_hdf(op, key='df', mode='r')

	return df

def create_csv_area_peak(filepath, arr, name, index=[]):
	op = os.path.join(filepath, 'peak_AUC_'+name+'.csv')
	df = pd.DataFrame(arr, index=index)
	
	df.to_csv(op)


# function to create dataframe for each event PSTH and save it to h5 file
def create_Df(filepath, event, name, psth, columns=[]):
	if name:
		op = os.path.join(filepath, event+'_{}.h5'.format(name))
	else:
		op = os.path.join(filepath, event+'.h5')
	
	# check if file already exists
	#if os.path.exists(op):
	#	return 0

	# removing psth binned trials
	columns = np.array(columns, dtype='str')
	regex = re.compile('bin_*') 
	single_trials = columns[[i for i in range(len(columns)) if not regex.match(columns[i])]]
	single_trials_index = [i for i in range(len(single_trials)) if single_trials[i]!='timestamps']

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
		df = pd.DataFrame(psth, index=None, columns=list(columns), dtype='float32')

	df.to_hdf(op, key='df', mode='w')


# function to read h5 file and make a dataframe from it
def read_Df(filepath, event, name):
	if name:
		op = os.path.join(filepath, event+'_{}.h5'.format(name))
	else:
		op = os.path.join(filepath, event+'.h5')
	df = pd.read_hdf(op, key='df', mode='r')

	return df


# function to create PSTH trials corresponding to each event timestamp
def rowFormation(z_score, thisIndex, nTsPrev, nTsPost):
    
	if nTsPrev<thisIndex and z_score.shape[0]>(thisIndex+nTsPost):
		res = z_score[thisIndex-nTsPrev-1:thisIndex+nTsPost]
	elif nTsPrev>=thisIndex and z_score.shape[0]>(thisIndex+nTsPost):
		mismatch = nTsPrev-thisIndex+1
		res = np.zeros(nTsPrev+nTsPost+1)
		res[:mismatch] = np.nan
		res[mismatch:] = z_score[:thisIndex+nTsPost]
	else:
		mismatch = (thisIndex+nTsPost)-z_score.shape[0]
		res1 = np.zeros(mismatch)
		res1[:] = np.nan
		res2 = z_score[thisIndex-nTsPrev-1:z_score.shape[0]]
		res = np.concatenate((res2, res1))

	return res


# function to calculate baseline for each PSTH trial and do baseline correction
def baselineCorrection(filepath, arr, timeAxis, baselineStart, baselineEnd):

	#timeAxis = read_Df(filepath, 'ts_psth', '')
	#timeAxis = np.asarray(timeAxis).reshape(-1)
	baselineStrtPt = np.where(timeAxis>=baselineStart)[0]
	baselineEndPt = np.where(timeAxis>=baselineEnd)[0]

	#print(baselineStrtPt[0], baselineEndPt[0])
	if baselineStart==0 and baselineEnd==0:
		return arr
	
	baseline = np.nanmean(arr[baselineStrtPt[0]:baselineEndPt[0]])
	baselineSub = np.subtract(arr, baseline)

	return baselineSub


# helper function to make PSTH for each event
def helper_psth(z_score, event, filepath, nSecPrev, nSecPost, timeInterval, bin_psth_trials, baselineStart, baselineEnd, naming, just_use_signal):

	sampling_rate = read_hdf5('timeCorrection_'+naming, filepath, 'sampling_rate')[0]

	# calculate time before event timestamp and time after event timestamp
	nTsPrev = int(round(nSecPrev*sampling_rate))
	nTsPost = int(round(nSecPost*sampling_rate))

	totalTs = (-1*nTsPrev) + nTsPost
	increment = ((-1*nSecPrev)+nSecPost)/totalTs
	timeAxis = np.linspace(nSecPrev, nSecPost+increment, totalTs+1)
	timeAxisNew = np.concatenate((timeAxis, timeAxis[::-1]))

	# avoid writing same data to same file in multi-processing
	#if not os.path.exists(os.path.join(filepath, 'ts_psth.h5')):
	#	print('file not exists')
	#	create_Df(filepath, 'ts_psth', '', timeAxis)
	#	time.sleep(2)

	ts = read_hdf5(event+'_'+naming, filepath, 'ts')
	
	# reject timestamps for which baseline cannot be calculated because of nan values
	new_ts = []
	for i in range(ts.shape[0]):
		thisTime = ts[i]-1
		if thisTime<abs(baselineStart):
			continue
		else:
			new_ts.append(ts[i])


	# reject burst of timestamps
	ts = np.asarray(new_ts)
	new_ts = [ts[0]]
	for i in range(1, ts.shape[0]):
		thisTime = ts[i]
		prevTime = new_ts[-1]
		diff = thisTime-prevTime
		if diff<timeInterval:
			continue
		else:
			new_ts.append(ts[i])

	# final timestamps
	ts = np.asarray(new_ts)
	nTs = ts.shape[0]

	# initialize PSTH vector
	psth = np.full((nTs, totalTs+1), np.nan)
	psth_baselineUncorrected = np.full((nTs, totalTs+1), np.nan)  # extra

	# for each timestamp, create trial which will be saved in a PSTH vector
	for i in range(nTs):
		thisTime = ts[i]    #-timeForLightsTurnOn
		thisIndex = int(round(thisTime*sampling_rate))
		arr = rowFormation(z_score, thisIndex, -1*nTsPrev, nTsPost)
		if just_use_signal==True:
			res = np.subtract(arr, np.nanmean(arr))
			z_score_arr = np.divide(res, np.nanstd(arr))
			arr = z_score_arr
		else:
			arr = arr
		
		psth_baselineUncorrected[i,:] = arr                                            # extra
		psth[i,:] = baselineCorrection(filepath, arr, timeAxis, baselineStart, baselineEnd)

	write_hdf5(ts, event+'_'+naming, filepath, 'ts')
	columns = list(ts)

	if bin_psth_trials>0:
		timestamps = read_hdf5('timeCorrection_'+naming, filepath, 'timestampNew')
		timestamps = np.divide(timestamps, 60)
		ts_min = np.divide(ts, 60)
		bin_steps = np.arange(timestamps[0], timestamps[-1]+bin_psth_trials, bin_psth_trials)

		psth_bin, psth_bin_baselineUncorrected = [], []
		
		for i in range(1, bin_steps.shape[0]):
			index = np.where((ts_min>=bin_steps[i-1]) & (ts_min<=bin_steps[i]))[0]

			# no trials in a given bin window, just put all the nan values
			if index.shape[0]==0:
				psth_bin.append(np.full(psth.shape[1], np.nan))
				psth_bin_baselineUncorrected.append(np.full(psth_baselineUncorrected.shape[1], np.nan))
				psth_bin.append(np.full(psth.shape[1], np.nan))
				psth_bin_baselineUncorrected.append(np.full(psth_baselineUncorrected.shape[1], np.nan))
			else:
				arr = psth[index,:]
				#  mean of bins
				psth_bin.append(np.nanmean(psth[index,:], axis=0))
				psth_bin_baselineUncorrected.append(np.nanmean(psth_baselineUncorrected[index,:], axis=0))
				psth_bin.append(np.nanstd(psth[index,:],axis=0)/math.sqrt(psth[index,:].shape[0]))
				# error of bins
				psth_bin_baselineUncorrected.append(np.nanstd(psth_baselineUncorrected[index,:],axis=0)/math.sqrt(psth_baselineUncorrected[index,:].shape[0]))
			
			# adding column names
			columns.append('bin_({}-{})'.format(np.around(bin_steps[i-1],0), np.around(bin_steps[i],0)))
			columns.append('bin_err_({}-{})'.format(np.around(bin_steps[i-1],0), np.around(bin_steps[i],0)))

		psth = np.concatenate((psth, psth_bin), axis=0)
		psth_baselineUncorrected = np.concatenate((psth_baselineUncorrected, psth_bin_baselineUncorrected), axis=0)

	timeAxis = timeAxis.reshape(1,-1)
	psth = np.concatenate((psth, timeAxis), axis=0)
	psth_baselineUncorrected = np.concatenate((psth_baselineUncorrected, timeAxis), axis=0)
	columns.append('timestamps')

	return psth, psth_baselineUncorrected, columns


# function to create PSTH for each event using function helper_psth and save the PSTH to h5 file
def storenamePsth(filepath, event, inputParameters):

	selectForComputePsth = inputParameters['selectForComputePsth']
	bin_psth_trials = inputParameters['bin_psth_trials']

	if selectForComputePsth=='z_score':
		path = glob.glob(os.path.join(filepath, 'z_score_*'))
	elif selectForComputePsth=='dff':
		path = glob.glob(os.path.join(filepath, 'dff_*'))
	else:
		path = glob.glob(os.path.join(filepath, 'z_score_*')) + glob.glob(os.path.join(filepath, 'dff_*'))

	b = np.divide(np.ones((100,)), 100)
	a = 1

	#storesList = storesList
	#sampling_rate = read_hdf5(storesList[0,0], filepath, 'sampling_rate')
	nSecPrev, nSecPost = inputParameters['nSecPrev'], inputParameters['nSecPost']
	baselineStart, baselineEnd = inputParameters['baselineCorrectionStart'], inputParameters['baselineCorrectionEnd']
	timeInterval = inputParameters['timeInterval']

	if 'control' in event.lower() or 'signal' in event.lower():
		return 0
	else:
		for i in range(len(path)):
			print("Computing PSTH for event {}...".format(event))
			basename = (os.path.basename(path[i])).split('.')[0]
			name_1 = basename.split('_')[-1]
			control = read_hdf5('control_'+name_1, os.path.dirname(path[i]), 'data')
			if (control==0).all()==True:
				signal = read_hdf5('signal_'+name_1, os.path.dirname(path[i]), 'data')
				z_score = ss.filtfilt(b, a, signal)
				just_use_signal = True
			else:
				z_score = read_hdf5('', path[i], 'data')
				just_use_signal = False
			psth, psth_baselineUncorrected, cols = helper_psth(z_score, event, filepath, 
															   nSecPrev, nSecPost, timeInterval, 
															   bin_psth_trials, 
															   baselineStart, baselineEnd, 
															   name_1, just_use_signal)

			create_Df(filepath, event+'_'+name_1+'_baselineUncorrected', basename, psth_baselineUncorrected, columns=cols)     # extra
			create_Df(filepath, event+'_'+name_1, basename, psth, columns=cols)

			print("PSTH for event {} computed.".format(event))


def helperPSTHPeakAndArea(psth_mean, timestamps, sampling_rate, peak_startPoint, peak_endPoint):

	peak_startPoint = np.asarray(peak_startPoint)
	peak_endPoint = np.asarray(peak_endPoint)

	peak_startPoint = peak_startPoint[~np.isnan(peak_startPoint)]
	peak_endPoint = peak_endPoint[~np.isnan(peak_endPoint)]

	#print(peak_startPoint, peak_endPoint)

	if peak_startPoint.shape[0]!=peak_endPoint.shape[0]:
		raise Exception('Number of Peak Start Time and Peak End Time are unequal.')

	if np.less_equal(peak_endPoint, peak_startPoint).any()==True:
		raise Exception('Peak End Time is lesser than or equal to Peak Start Time. Please check the Peak parameters window.')

	#print(peak_startPoint, peak_endPoint)

	peak_area = OrderedDict()

	if peak_startPoint.shape[0]==0 or peak_endPoint.shape[0]==0:
		peak_area['peak'] = np.nan
		peak_area['area'] = np.nan

	for i in range(peak_startPoint.shape[0]):
		startPtForPeak = np.where(timestamps>=peak_startPoint[i])[0]
		endPtForPeak = np.where(timestamps>=peak_endPoint[i])[0]
		if len(startPtForPeak)>=1 and len(endPtForPeak)>=1:
			peakPoint_pos = startPtForPeak[0] + np.argmax(psth_mean[startPtForPeak[0]:endPtForPeak[0],:], axis=0)
			peakPoint_neg = startPtForPeak[0] + np.argmin(psth_mean[startPtForPeak[0]:endPtForPeak[0],:], axis=0)
			peak_area['peak_pos_'+str(i+1)] = np.amax(psth_mean[peakPoint_pos],axis=0)
			peak_area['peak_neg_'+str(i+1)] = np.amin(psth_mean[peakPoint_neg],axis=0)
			peak_area['area_'+str(i+1)] = np.trapz(psth_mean[startPtForPeak[0]:endPtForPeak[0],:], axis=0)
		else:
			peak_area['peak_'+str(i+1)] = np.nan
			peak_area['area_'+str(i+1)] = np.nan

	return peak_area


# function to compute PSTH peak and area using the function helperPSTHPeakAndArea save the values to h5 and csv files.
def findPSTHPeakAndArea(filepath, event, inputParameters):

	#sampling_rate = read_hdf5(storesList[0,0], filepath, 'sampling_rate')
	peak_startPoint = inputParameters['peak_startPoint']
	peak_endPoint = inputParameters['peak_endPoint']
	selectForComputePsth = inputParameters['selectForComputePsth']


	if selectForComputePsth=='z_score':
		path = glob.glob(os.path.join(filepath, 'z_score_*'))
	elif selectForComputePsth=='dff':
		path = glob.glob(os.path.join(filepath, 'dff_*'))
	else:
		path = glob.glob(os.path.join(filepath, 'z_score_*')) + glob.glob(os.path.join(filepath, 'dff_*'))


	if 'control' in event.lower() or 'signal' in event.lower():
		return 0
	else:
		for i in range(len(path)):
			print('Computing peak and area for PSTH mean signal for event {}...'.format(event))
			basename = (os.path.basename(path[i])).split('.')[0]
			name_1 = basename.split('_')[-1]
			sampling_rate = read_hdf5('timeCorrection_'+name_1, filepath, 'sampling_rate')[0]
			psth = read_Df(filepath, event+'_'+name_1, basename)
			cols = list(psth.columns)
			regex = re.compile('bin_[(]')
			bin_names = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
			regex_trials = re.compile('[+-]?([0-9]*[.])?[0-9]+')
			trials_names = [cols[i] for i in range(len(cols)) if regex_trials.match(cols[i])]
			psth_mean_bin_names = trials_names + bin_names + ['mean']
			psth_mean_bin_mean = np.asarray(psth[psth_mean_bin_names])
			timestamps = np.asarray(psth['timestamps']).ravel() #np.asarray(read_Df(filepath, 'ts_psth', '')).ravel()
			peak_area = helperPSTHPeakAndArea(psth_mean_bin_mean, timestamps, sampling_rate, peak_startPoint, peak_endPoint)   # peak, area = 
			#arr = np.array([[peak, area]])
			fileName = [os.path.basename(os.path.dirname(filepath))]
			index = [fileName[0]+'_'+s for s in psth_mean_bin_names]
			create_Df_area_peak(filepath, peak_area, event+'_'+name_1+'_'+basename, index=index) # columns=['peak', 'area']
			create_csv_area_peak(filepath, peak_area, event+'_'+name_1+'_'+basename, index=index)

			print('Peak and Area for PSTH mean signal for event {} computed.'.format(event))

def makeAverageDir(filepath):

	op = os.path.join(filepath, 'average')
	if not os.path.exists(op):
		os.mkdir(op)

	return op

def psth_shape_check(psth):

	each_ln = []
	for i in range(len(psth)):
		each_ln.append(psth[i].shape[0])

	each_ln = np.asarray(each_ln)
	keep_ln = each_ln[-1]

	for i in range(len(psth)):
		if psth[i].shape[0]>keep_ln:
			psth[i] = psth[i][:keep_ln]
		elif psth[i].shape[0]<keep_ln:
			psth[i] = np.append(psth[i], np.full(keep_ln-len(psth[i]), np.nan))
		else:
			psth[i] = psth[i]

	return psth

# function to compute average of group of recordings
def averageForGroup(folderNames, event, inputParameters):

	print("Averaging group of data...")

	path = []
	abspath = inputParameters['abspath']
	selectForComputePsth = inputParameters['selectForComputePsth']
	path_temp_len = []
	op = makeAverageDir(abspath)

	# combining paths to all the selected folders for doing average
	for i in range(len(folderNames)):
		if selectForComputePsth=='z_score':
			path_temp = glob.glob(os.path.join(folderNames[i], 'z_score_*'))
		elif selectForComputePsth=='dff':
			path_temp = glob.glob(os.path.join(folderNames[i], 'dff_*'))
		else:
			path_temp = glob.glob(os.path.join(folderNames[i], 'z_score_*')) + glob.glob(os.path.join(folderNames[i], 'dff_*'))

		path_temp_len.append(len(path_temp))
		#path_temp = glob.glob(os.path.join(folderNames[i], 'z_score_*'))
		for j in range(len(path_temp)):
			basename = (os.path.basename(path_temp[j])).split('.')[0]
			write_hdf5(np.array([]), basename, op, 'data')
			name_1 = basename.split('_')[-1]
			temp = [folderNames[i], event+'_'+name_1, basename]
			path.append(temp)

	# processing of all the paths
	path_temp_len = np.asarray(path_temp_len)
	max_len = np.argmax(path_temp_len)
	
	naming = []
	for i in range(len(path)):
		naming.append(path[i][2])
	naming = np.unique(np.asarray(naming))

	new_path = [[] for _ in range(path_temp_len[max_len])]
	for i in range(len(path)):
		idx = np.where(naming==path[i][2])[0][0]
		new_path[idx].append(path[i])

	# read PSTH for each event and make the average of it. Save the final output to an average folder.
	for i in range(len(new_path)):
		psth, psth_bins = [], [] 
		columns = []
		temp_path = new_path[i]
		for j in range(len(temp_path)):
			#print(os.path.join(temp_path[j][0], temp_path[j][1]+'_{}.h5'.format(temp_path[j][2])))
			if not os.path.exists(os.path.join(temp_path[j][0], temp_path[j][1]+'_{}.h5'.format(temp_path[j][2]))):
				continue
			else:
				df = read_Df(temp_path[j][0], temp_path[j][1], temp_path[j][2])    # filepath, event, name
				cols = list(df.columns)
				regex = re.compile('bin_[(]')
				bins_cols = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
				psth.append(np.asarray(df['mean']))
				columns.append(os.path.basename(temp_path[j][0]))
				if len(bins_cols)>0:
					psth_bins.append(df[bins_cols])
					
		if len(bins_cols)>0:
			df_bins = pd.concat(psth_bins, axis=1)
			df_bins_mean = df_bins.groupby(by=df_bins.columns, axis=1).mean()
			df_bins_err = df_bins.groupby(by=df_bins.columns, axis=1).std()/math.sqrt(df_bins.shape[1]) 
			cols_err = list(df_bins_err.columns)
			dict_err = {}
			for i in cols_err:
				split = i.split('_')
				dict_err[i] = '{}_err_{}'.format(split[0], split[1])
			df_bins_err = df_bins_err.rename(columns=dict_err)
			columns = columns + list(df_bins_mean.columns) + list(df_bins_err.columns)
			df_bins_mean_err = pd.concat([df_bins_mean, df_bins_err], axis=1).T
			psth, df_bins_mean_err = np.asarray(psth), np.asarray(df_bins_mean_err)
			psth = np.concatenate((psth, df_bins_mean_err), axis=0)
		else:
			psth = psth_shape_check(psth)
			psth = np.asarray(psth)

		timestamps = np.asarray(df['timestamps']).reshape(1,-1)
		psth = np.concatenate((psth, timestamps), axis=0)
		columns = columns + ['timestamps']
		create_Df(op, temp_path[j][1], temp_path[j][2], psth, columns=columns)


	# read PSTH peak and area for each event and combine them. Save the final output to an average folder
	for i in range(len(new_path)):
		arr = [] 
		index = []
		temp_path = new_path[i]
		for j in range(len(temp_path)):
			if not os.path.exists(os.path.join(temp_path[j][0], 'peak_AUC_'+temp_path[j][1]+'_'+temp_path[j][2]+'.h5')):
				continue
			else:
				df = read_Df_area_peak(temp_path[j][0], temp_path[j][1]+'_'+temp_path[j][2])
				arr.append(df)
				index.append(list(df.index))
		
		index = list(np.concatenate(index))
		new_df = pd.concat(arr, axis=0)  #os.path.join(filepath, 'peak_AUC_'+name+'.csv')
		new_df.to_csv(os.path.join(op, 'peak_AUC_{}_{}.csv'.format(temp_path[j][1], temp_path[j][2])), index=index)
		new_df.to_hdf(os.path.join(op, 'peak_AUC_{}_{}.h5'.format(temp_path[j][1], temp_path[j][2])), key='df', mode='w', index=index)

	print("Group of data averaged.")


def psthForEachStorename(inputParametersPath):

	print("Computing PSTH, Peak and Area for each event...")

	with open(inputParametersPath) as f:	
		inputParameters = json.load(f)


	#storesList = np.genfromtxt(inputParameters['storesListPath'], dtype='str', delimiter=',')

	folderNames = inputParameters['folderNames']
	folderNamesForAvg = inputParameters['folderNamesForAvg']
	average = inputParameters['averageForGroup']
	combine_data = inputParameters['combine_data']

	print("Average for group : ", average)

	# for average following if statement will be executed
	if average==True:
		if len(folderNamesForAvg)>0:
			storesListPath = []
			for i in range(len(folderNamesForAvg)):
				filepath = folderNamesForAvg[i]
				storesListPath.append(glob.glob(os.path.join(filepath, '*_output_*')))
			storesListPath = np.concatenate(storesListPath)
			storesList = np.asarray([[],[]])
			for i in range(storesListPath.shape[0]):
				storesList = np.concatenate((storesList, np.genfromtxt(os.path.join(storesListPath[i], 'storesList.csv'), dtype='str', delimiter=',')), axis=1)
			storesList = np.unique(storesList, axis=1)
			op = makeAverageDir(inputParameters['abspath'])
			np.savetxt(os.path.join(op, 'storesList.csv'), storesList, delimiter=",", fmt='%s')
			for k in range(storesList.shape[1]):
				
				if 'control' in storesList[1,k].lower() or 'signal' in storesList[1,k].lower():
					continue
				else:
					averageForGroup(storesListPath, storesList[1,k], inputParameters)

		else:
			raise Exception('Not a single folder name is provided in folderNamesForAvg in inputParamters File.')

	# for individual analysis following else statement will be executed
	else:
		if combine_data==True:
			storesListPath = []
			for i in range(len(folderNames)):
				filepath = folderNames[i]
				storesListPath.append(glob.glob(os.path.join(filepath, '*_output_*')))
			storesListPath = list(np.concatenate(storesListPath).flatten())
			op = get_all_stores_for_combining_data(storesListPath)
			for i in range(len(op)):
				storesList = np.asarray([[],[]])
				for j in range(len(op[i])):
					storesList = np.concatenate((storesList, np.genfromtxt(os.path.join(op[i][j], 'storesList.csv'), dtype='str', delimiter=',')), axis=1)
				storesList = np.unique(storesList, axis=1)
				for k in range(storesList.shape[1]):
					storenamePsth(op[i][0], storesList[1,k], inputParameters)
					findPSTHPeakAndArea(op[i][0], storesList[1,k], inputParameters)
		else:
			for i in range(len(folderNames)):
				filepath = folderNames[i]
				storesListPath = glob.glob(os.path.join(filepath, '*_output_*'))
				for j in range(len(storesListPath)):
					filepath = storesListPath[j]
					storesList = np.genfromtxt(os.path.join(filepath, 'storesList.csv'), dtype='str', delimiter=',')

					with mp.Pool(mp.cpu_count()) as p:
						p.starmap(storenamePsth, zip(repeat(filepath), storesList[1,:], repeat(inputParameters)))

					with mp.Pool(mp.cpu_count()) as pq:
						pq.starmap(findPSTHPeakAndArea, zip(repeat(filepath), storesList[1,:], repeat(inputParameters)))

					#for k in range(storesList.shape[1]):
					#	storenamePsth(filepath, storesList[1,k], inputParameters)
					#	findPSTHPeakAndArea(filepath, storesList[1,k], inputParameters)


	print("PSTH, Area and Peak are computed for all events.")

if __name__ == "__main__":
	psthForEachStorename(sys.argv[1:][0])

