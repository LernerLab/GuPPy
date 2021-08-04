import os
import json
import glob
import numpy as np 
import h5py
import math
import pandas as pd
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
	
	psth = psth.T
	if psth.ndim > 1:
		mean = np.nanmean(psth, axis=1).reshape(-1,1)
		err = np.nanstd(psth, axis=1)/math.sqrt(psth.shape[1])
		err = err.reshape(-1,1)
		psth = np.hstack((psth,mean))
		psth = np.hstack((psth, err))
		timestamps = np.asarray(read_Df(filepath, 'ts_psth', ''))
		psth = np.hstack((psth, timestamps))
	try:
		ts = read_hdf5(event, filepath, 'ts')
		ts = np.append(ts, ['mean', 'err', 'timestamps'])
	except:
		ts = None

	if len(columns)==0:
		df = pd.DataFrame(psth, index=None, columns=ts, dtype='float32')
	else:
		columns = np.asarray(columns)
		columns = np.append(columns, ['mean', 'err', 'timestamps'])
		df = pd.DataFrame(psth, index=None, columns=columns, dtype='float32')

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
def baselineCorrection(filepath, arr, baselineStart, baselineEnd):

	timeAxis = read_Df(filepath, 'ts_psth', '')
	timeAxis = np.asarray(timeAxis).reshape(-1)
	baselineStrtPt = np.where(timeAxis>=baselineStart)[0]
	baselineEndPt = np.where(timeAxis>=baselineEnd)[0]

	#print(baselineStrtPt[0], baselineEndPt[0])
	if baselineStart==0 and baselineEnd==0:
		return arr
	
	baseline = np.nanmean(arr[baselineStrtPt[0]:baselineEndPt[0]])
	baselineSub = np.subtract(arr, baseline)

	return baselineSub


# helper function to make PSTH for each event
def helper_psth(z_score, event, filepath, nSecPrev, nSecPost, timeInterval, baselineStart, baselineEnd, naming, just_use_signal):

	sampling_rate = read_hdf5('timeCorrection_'+naming, filepath, 'sampling_rate')[0]

	# calculate time before event timestamp and time after event timestamp
	nTsPrev = int(round(nSecPrev*sampling_rate))
	nTsPost = int(round(nSecPost*sampling_rate))

	totalTs = (-1*nTsPrev) + nTsPost
	increment = ((-1*nSecPrev)+nSecPost)/totalTs
	timeAxis = np.arange(nSecPrev, nSecPost+increment, increment) # change -1*nSecPrev
	timeAxisNew = np.concatenate((timeAxis, timeAxis[::-1]))

	create_Df(filepath, 'ts_psth', '', timeAxis)
	create_Df(filepath, 'ts_new_psth', '', timeAxisNew)

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
		psth[i,:] = baselineCorrection(filepath, arr, baselineStart, baselineEnd)

	write_hdf5(ts, event+'_'+naming, filepath, 'ts')

	return psth, psth_baselineUncorrected


# function to create PSTH for each event using function helper_psth and save the PSTH to h5 file
def storenamePsth(filepath, event, inputParameters, storesList):

	selectForComputePsth = inputParameters['selectForComputePsth']
	if selectForComputePsth=='z_score':
		path = glob.glob(os.path.join(filepath, 'z_score_*'))
	elif selectForComputePsth=='dff':
		path = glob.glob(os.path.join(filepath, 'dff_*'))
	else:
		path = glob.glob(os.path.join(filepath, 'z_score_*')) + glob.glob(os.path.join(filepath, 'dff_*'))

	b = np.divide(np.ones((100,)), 100)
	a = 1

	storesList = storesList
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
			psth, psth_baselineUncorrected = helper_psth(z_score, event, filepath, nSecPrev, nSecPost, timeInterval, baselineStart, baselineEnd, name_1, just_use_signal)
			create_Df(filepath, event+'_'+name_1+'_baselineUncorrected', basename, psth_baselineUncorrected)     # extra
			create_Df(filepath, event+'_'+name_1, basename, psth)

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
			peakPoint = startPtForPeak[0] + np.argmax(psth_mean[startPtForPeak[0]:endPtForPeak[0]])
			peak_area['peak_'+str(i+1)] = psth_mean[peakPoint]

			#for j in range(startPtForPeak[0], endPtForPeak[0]):
			#	arr = psth_mean[j:int(j+(0.5*sampling_rate))]
			#	if psth_mean[j]<0 and (arr<0).all():
			#		areaEndPt = j
			#		break
			#if j==endPtForPeak[0]-1:
			#	areaEndPt = endPtForPeak[0]-1

			peak_area['area_'+str(i+1)] = np.trapz(psth_mean[startPtForPeak[0]:endPtForPeak[0]])
		else:
			peak_area['peak_'+str(i+1)] = np.nan
			peak_area['area_'+str(i+1)] = np.nan

	#print(peak_area)

	return peak_area


# function to compute PSTH peak and area using the function helperPSTHPeakAndArea save the values to h5 and csv files.
def findPSTHPeakAndArea(filepath, event, inputParameters, storesList):


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
			psth_mean = np.asarray(psth['mean'])
			timestamps = np.asarray(read_Df(filepath, 'ts_psth', '')).ravel()
			peak_area = helperPSTHPeakAndArea(psth_mean, timestamps, sampling_rate, peak_startPoint, peak_endPoint)   # peak, area = 
			#arr = np.array([[peak, area]])
			fileName = [os.path.basename(os.path.dirname(filepath))]
			create_Df_area_peak(filepath, peak_area, event+'_'+name_1+'_'+basename, index=fileName) # columns=['peak', 'area']
			create_csv_area_peak(filepath, peak_area, event+'_'+name_1+'_'+basename, index=fileName)

			print('Peak and Area for PSTH mean signal for event {} computed.'.format(event))

def makeAverageDir(filepath):

	op = os.path.join(filepath, 'average')
	if not os.path.exists(op):
		os.mkdir(op)

	return op

# function to compute average of group of recordings
def averageForGroup(folderNames, event, inputParameters):

	print("Averaging group of data...")

	path = []
	abspath = inputParameters['abspath']
	selectForComputePsth = inputParameters['selectForComputePsth']
	path_temp_len = []

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



	timestamps = np.asarray(read_Df(new_path[0][0][0], 'ts_psth', '')).reshape(-1)
	op = makeAverageDir(abspath)
	create_Df(op, 'ts_psth', '', timestamps)

	# read PSTH for each event and make the average of it. Save the final output to an average folder.
	for i in range(len(new_path)):
		psth = [] 
		columns = []
		temp_path = new_path[i]
		for j in range(len(temp_path)):
			#print(os.path.join(temp_path[j][0], temp_path[j][1]+'_{}.h5'.format(temp_path[j][2])))
			if not os.path.exists(os.path.join(temp_path[j][0], temp_path[j][1]+'_{}.h5'.format(temp_path[j][2]))):
				continue
			else:
				df = read_Df(temp_path[j][0], temp_path[j][1], temp_path[j][2])    # filepath, event, name
				psth.append(np.asarray(df['mean']))
				columns.append(os.path.basename(temp_path[j][0]))

		psth = np.asarray(psth)
		create_Df(op, temp_path[j][1], temp_path[j][2], psth, columns=columns)


	# read PSTH peak and area for each event and combine them. Save the final output to an average folder
	for i in range(len(new_path)):
		arr = [] 
		fileName = []
		temp_path = new_path[i]
		for j in range(len(temp_path)):
			if not os.path.exists(os.path.join(temp_path[j][0], 'peak_AUC_'+temp_path[j][1]+'_'+temp_path[j][2]+'.h5')):
				continue
			else:
				df = read_Df_area_peak(temp_path[j][0], temp_path[j][1]+'_'+temp_path[j][2])
				arr.append(df)
				fileName.append(os.path.basename(temp_path[j][0]))
		
		new_df = pd.concat(arr, axis=0)  #os.path.join(filepath, 'peak_AUC_'+name+'.csv')
		new_df.to_csv(os.path.join(op, 'peak_AUC_{}_{}.csv'.format(temp_path[j][1], temp_path[j][2])), index=fileName)
		new_df.to_hdf(os.path.join(op, 'peak_AUC_{}_{}.h5'.format(temp_path[j][1], temp_path[j][2])), key='df', mode='w', index=fileName)
		#create_Df_area_peak(op, new_df, temp_path[j][1]+'_'+temp_path[j][2], index=fileName)
		#create_csv_area_peak(op, new_df, temp_path[j][1]+'_'+temp_path[j][2], index=fileName)

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
					storenamePsth(op[i][0], storesList[1,k], inputParameters, storesList)
					findPSTHPeakAndArea(op[i][0], storesList[1,k], inputParameters, storesList)
		else:
			for i in range(len(folderNames)):
				filepath = folderNames[i]
				storesListPath = glob.glob(os.path.join(filepath, '*_output_*'))
				for j in range(len(storesListPath)):
					filepath = storesListPath[j]
					storesList = np.genfromtxt(os.path.join(filepath, 'storesList.csv'), dtype='str', delimiter=',')
					for k in range(storesList.shape[1]):
						storenamePsth(filepath, storesList[1,k], inputParameters, storesList)
						findPSTHPeakAndArea(filepath, storesList[1,k], inputParameters, storesList)


	print("PSTH, Area and Peak are computed for all events.")




