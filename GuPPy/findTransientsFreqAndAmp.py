import os
import sys
import glob
import h5py
import json
import math
import numpy as np 
import pandas as pd 
import multiprocessing as mp
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from preprocess import get_all_stores_for_combining_data


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


def processChunks(arrValues, arrIndexes):
    
    arrValues = arrValues[~np.isnan(arrValues)] 
    median = np.median(arrValues)			
    										
    mad = np.median(np.abs(arrValues-median))   
    										
    firstThreshold = median + (2*mad)
    										
    										
    greaterThanMad = np.where(arrValues>firstThreshold)[0]
    

    arr = np.arange(arrValues.shape[0])
    lowerThanMad = np.isin(arr, greaterThanMad, invert=True)
    filteredOut = arrValues[np.where(lowerThanMad==True)[0]]
    
    filteredOutMedian = np.median(filteredOut)
    filteredOutMad = np.median(np.abs(filteredOut-np.median(filteredOut)))
    secondThreshold = filteredOutMedian+(3*filteredOutMad)

    greaterThanThreshIndex = np.where(arrValues>secondThreshold)[0]
    greaterThanThreshValues = arrValues[greaterThanThreshIndex]
    temp = np.zeros(arrValues.shape[0])
    temp[greaterThanThreshIndex] = greaterThanThreshValues
    peaks = argrelextrema(temp, np.greater)[0]

    firstThresholdY = np.full(arrValues.shape[0], firstThreshold)
    secondThresholdY = np.full(arrValues.shape[0], secondThreshold)


    newPeaks = np.full(arrValues.shape[0], np.nan)
    newPeaks[peaks] = peaks + arrIndexes[0]

    #madY = np.full(arrValues.shape[0], mad)
    medianY = np.full(arrValues.shape[0], median)
    filteredOutMedianY = np.full(arrValues.shape[0], filteredOutMedian)

    return peaks, mad, filteredOutMad, medianY, filteredOutMedianY, firstThresholdY, secondThresholdY



def createChunks(z_score, sampling_rate, window):
	
	print('Creating chunks for multiprocessing...')

	windowPoints = math.ceil(sampling_rate*window)
	remainderPoints = math.ceil((sampling_rate*window) - (z_score.shape[0]%windowPoints))


	if remainderPoints==windowPoints:
		padded_z_score = z_score
		z_score_index = np.arange(padded_z_score.shape[0])
	else:
	    padding = np.full(remainderPoints, np.nan)
	    padded_z_score = np.concatenate((z_score, padding))
	    z_score_index = np.arange(padded_z_score.shape[0])

	reshape = padded_z_score.shape[0]/windowPoints

	if reshape.is_integer()==True:
	    z_score_chunks = padded_z_score.reshape(int(reshape), -1)
	    z_score_chunks_index = z_score_index.reshape(int(reshape), -1)
	else:
	    raise Exception('Reshaping values should be integer.')

	print('Chunks are created for multiprocessing.')
	return z_score_chunks, z_score_chunks_index


def calculate_freq_amp(arr, z_score, z_score_chunks_index, timestamps):
	peaks = arr[:,0]
	filteredOutMedian = arr[:,4]
	count = 0
	peaksAmp = np.array([])
	peaksInd = np.array([])
	for i in range(z_score_chunks_index.shape[0]):
	    count += peaks[i].shape[0]
	    peaksIndexes = peaks[i]+z_score_chunks_index[i][0]
	    peaksInd = np.concatenate((peaksInd, peaksIndexes))
	    amps = z_score[peaksIndexes]-filteredOutMedian[i][0]
	    peaksAmp = np.concatenate((peaksAmp, amps))

	peaksInd = peaksInd.ravel()
	peaksInd = peaksInd.astype(int)
	#print(timestamps)
	freq = peaksAmp.shape[0]/((timestamps[-1]-timestamps[0])/60)

	return freq, peaksAmp, peaksInd

def create_Df(filepath, arr, name, index=[], columns=[]):

	op = os.path.join(filepath, 'freqAndAmp_'+name+'.h5')
	dirname = os.path.dirname(filepath)

	df = pd.DataFrame(arr, index=index, columns=columns)

	df.to_hdf(op, key='df', mode='w')

def create_csv(filepath, arr, name, index=[], columns=[]):
	op = os.path.join(filepath, name)
	df = pd.DataFrame(arr, index=index, columns=columns)
	df.to_csv(op)

def read_Df(filepath, name):
	op = os.path.join(filepath, 'freqAndAmp_'+name+'.h5')
	df = pd.read_hdf(op, key='df', mode='r')

	return df

def visuzlize_peaks(filepath, z_score, timestamps, peaksIndex):

	dirname = os.path.dirname(filepath)

	basename = (os.path.basename(filepath)).split('.')[0]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(timestamps,z_score, '-',
			timestamps[peaksIndex], z_score[peaksIndex], 'o')
	ax.set_title(basename)
	fig.suptitle(os.path.basename(dirname))
	#plt.show()

def findFreqAndAmp(filepath, inputParameters, window=15):

	print('calculating frequency and amplitude of transients in z-score data....')
	selectForTransientsComputation = inputParameters['selectForTransientsComputation']

	if selectForTransientsComputation=='z_score':
		path = glob.glob(os.path.join(filepath, 'z_score_*'))
	elif selectForTransientsComputation=='dff':
		path = glob.glob(os.path.join(filepath, 'dff_*'))
	else:
		path = glob.glob(os.path.join(filepath, 'z_score_*')) + glob.glob(os.path.join(filepath, 'dff_*'))

	for i in range(len(path)):
		basename = (os.path.basename(path[i])).split('.')[0]
		name_1 = basename.split('_')[-1]
		sampling_rate = read_hdf5('timeCorrection_'+name_1, filepath, 'sampling_rate')[0]
		z_score = read_hdf5('', path[i], 'data')
		z_score_chunks, z_score_chunks_index = createChunks(z_score, sampling_rate, window)

		#p = mp.Pool(mp.cpu_count())
		with mp.Pool(mp.cpu_count()) as p:
			result = p.starmap(processChunks, zip(z_score_chunks, z_score_chunks_index))
		#p.close()
		#p.join()
		
		result = np.asarray(result, dtype=object)
		ts = read_hdf5('timeCorrection_'+name_1, filepath, 'timestampNew')
		freq, peaksAmp, peaksInd = calculate_freq_amp(result, z_score, z_score_chunks_index, ts)
		peaks_occurrences = np.array([ts[peaksInd], peaksAmp]).T
		arr = np.array([[freq, np.mean(peaksAmp)]])
		fileName = [os.path.basename(os.path.dirname(filepath))]
		create_Df(filepath, arr, basename, index=fileName ,columns=['freq (events/min)', 'amplitude'])
		create_csv(filepath, arr, 'freqAndAmp_'+basename+'.csv', 
				   index=fileName, columns=['freq (events/min)', 'amplitude'])
		create_csv(filepath, peaks_occurrences, 'transientsOccurrences_'+basename+'.csv', 
				   index=np.arange(peaks_occurrences.shape[0]),columns=['timestamps', 'amplitude'])
		visuzlize_peaks(path[i], z_score, ts, peaksInd)
	
	print('Frequency and amplitude of transients in z_score data are calculated.')
		


def makeAverageDir(filepath):

	op = os.path.join(filepath, 'average')
	if not os.path.exists(op):
		os.mkdir(op)

	return op	

def averageForGroup(folderNames, inputParameters):

	print('Combining results for frequency and amplitude of transients in z-score data...')
	path = []
	abspath = inputParameters['abspath']
	selectForTransientsComputation = inputParameters['selectForTransientsComputation']
	path_temp_len = []

	for i in range(len(folderNames)):
		if selectForTransientsComputation=='z_score':
			path_temp = glob.glob(os.path.join(folderNames[i], 'z_score_*')) 
		elif selectForTransientsComputation=='dff':
			path_temp = glob.glob(os.path.join(folderNames[i], 'dff_*'))
		else:
			path_temp = glob.glob(os.path.join(folderNames[i], 'z_score_*')) + glob.glob(os.path.join(folderNames[i], 'dff_*'))

		path_temp_len.append(len(path_temp))

		for j in range(len(path_temp)):
			basename = (os.path.basename(path_temp[j])).split('.')[0]
			#name = name[0]
			temp = [folderNames[i], basename]
			path.append(temp)


	path_temp_len = np.asarray(path_temp_len)
	max_len = np.argmax(path_temp_len)

	naming = []
	for i in range(len(path)):
		naming.append(path[i][1])
	naming = np.unique(np.asarray(naming))
	

	new_path = [[] for _ in range(path_temp_len[max_len])]
	for i in range(len(path)):
		idx = np.where(naming==path[i][1])[0][0]
		new_path[idx].append(path[i])

	op = makeAverageDir(abspath)

	
	for i in range(len(new_path)):
		arr = [] #np.zeros((len(new_path[i]), 2))
		fileName = []
		temp_path = new_path[i]
		for j in range(len(temp_path)):
			if not os.path.exists(os.path.join(temp_path[j][0], 'freqAndAmp_'+temp_path[j][1]+'.h5')):
				continue
			else:
				df = read_Df(temp_path[j][0], temp_path[j][1])
				arr.append(np.array([df['freq (events/min)'][0], df['amplitude'][0]]))
				fileName.append(os.path.basename(temp_path[j][0]))

		arr = np.asarray(arr)
		create_Df(op, arr, temp_path[j][1], index=fileName, columns=['freq (events/min)', 'amplitude'])
		create_csv(op, arr, 'freqAndAmp_'+temp_path[j][1]+'.csv', index=fileName, columns=['freq (events/min)', 'amplitude'])

	print('Results for frequency and amplitude of transients in z-score data are combined.')

def executeFindFreqAndAmp(inputParametersPath):

	print('Finding transients in z-score data and calculating frequency and amplitude....')
	with open(inputParametersPath) as f:	
		inputParameters = json.load(f)

	average = inputParameters['averageForGroup']
	folderNamesForAvg = inputParameters['folderNamesForAvg']
	folderNames = inputParameters['folderNames']
	combine_data = inputParameters['combine_data']
	moving_window = inputParameters['moving_window']

	if average==True:
		if len(folderNamesForAvg)>0:
			storesListPath = []
			for i in range(len(folderNamesForAvg)):
				filepath = folderNamesForAvg[i]
				storesListPath.append(glob.glob(os.path.join(filepath, '*_output_*')))
			storesListPath = np.concatenate(storesListPath)
			averageForGroup(storesListPath, inputParameters)
		else:
			raise Exception('Not a single folder name is provided in folderNamesForAvg in inputParamters File.')
			
			
	else:
		if combine_data==True:
			storesListPath = []
			for i in range(len(folderNames)):
				filepath = folderNames[i]
				storesListPath.append(glob.glob(os.path.join(filepath, '*_output_*')))
			storesListPath = list(np.concatenate(storesListPath).flatten())
			op = get_all_stores_for_combining_data(storesListPath)
			for i in range(len(op)):
				filepath = op[i][0]
				storesList = np.genfromtxt(os.path.join(filepath, 'storesList.csv'), dtype='str', delimiter=',')
				findFreqAndAmp(filepath, inputParameters, window=moving_window)
			plt.show()
		else:
			for i in range(len(folderNames)):
				filepath = folderNames[i]
				storesListPath = glob.glob(os.path.join(filepath, '*_output_*'))
				for j in range(len(storesListPath)):
					filepath = storesListPath[j]
					storesList = np.genfromtxt(os.path.join(filepath, 'storesList.csv'), dtype='str', delimiter=',')
					findFreqAndAmp(filepath, inputParameters, window=moving_window)
			plt.show()

	print('Transients in z-score data found and frequency and amplitude are calculated.')


if __name__ == "__main__":
	executeFindFreqAndAmp(sys.argv[1:][0])


