import os
import json
import glob
import time
import re
import fnmatch
import numpy as np 
import h5py
import math
from scipy import signal as ss
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
from combineDataFn import processTimestampsForCombiningData


# find files by ignoreing the case sensitivity
def find_files(path, glob_path, ignore_case = False):
    rule = re.compile(fnmatch.translate(glob_path), re.IGNORECASE) if ignore_case \
            else re.compile(fnmatch.translate(glob_path))
    return [os.path.join(path,n) for n in os.listdir(os.path.expanduser(path)) if rule.match(n)]


# function to read hdf5 file
def read_hdf5(event, filepath, key):
	if event:
		event = event.replace("\\","_")
		event = event.replace("/","_")
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
				f.create_dataset(key, data=data, maxshape=(None,), chunks=True)
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
				if type(data) is np.ndarray:
					f.create_dataset(key, data=data, maxshape=(None,), chunks=True)
				else:
					f.create_dataset(key, data=data)


# function to correct timestamps after eliminating fist few seconds of the data
def timestampCorrection(filepath, timeForLightsTurnOn, storesList):

	print("Correcting timestamps by getting rid of the first {} seconds and convert timestamps to seconds...".format(timeForLightsTurnOn))
	storenames = storesList[0,:]
	storesList = storesList[1,:]
	
	arr = []
	for i in range(storesList.shape[0]):
	    if 'control' in storesList[i].lower() or 'signal' in storesList[i].lower():
	        arr.append(storesList[i])

	arr = sorted(arr)
	try:
		arr = np.asarray(arr).reshape(2,-1)
	except:
		raise Exception('Error in saving stores list file or spelling mistake for control or signal')

	for i in range(arr.shape[1]):
		name_1 = arr[0,i].split('_')[-1]
		name_2 = arr[1,i].split('_')[-1]
		#dirname = os.path.dirname(path[i])
		idx = np.where(storesList==arr[1,i])[0]

		if idx.shape[0]==0:
			raise Exception('{} does not exist in the stores list file.'.format(arr[0,i]))
		
		timestamp = read_hdf5(storenames[idx][0], filepath, 'timestamps')
		npoints = read_hdf5(storenames[idx][0], filepath, 'npoints')
		sampling_rate = read_hdf5(storenames[idx][0], filepath, 'sampling_rate')
		
		if name_1==name_2:
			timeRecStart = timestamp[0]
			timestamps = np.subtract(timestamp, timeRecStart)
			adder = np.arange(npoints)/sampling_rate
			lengthAdder = adder.shape[0]
			timestampNew = np.zeros((len(timestamps), lengthAdder))
			for i in range(lengthAdder):
			    timestampNew[:,i] = np.add(timestamps, adder[i])
			timestampNew = (timestampNew.T).reshape(-1, order='F')
			correctionIndex = np.where(timestampNew>=timeForLightsTurnOn)[0]
			timestampNew = timestampNew[correctionIndex]

			write_hdf5(np.asarray([timeRecStart]), 'timeCorrection_'+name_1, filepath, 'timeRecStart')
			write_hdf5(timestampNew, 'timeCorrection_'+name_1, filepath, 'timestampNew')
			write_hdf5(correctionIndex, 'timeCorrection_'+name_1, filepath, 'correctionIndex')
			write_hdf5(np.asarray([sampling_rate]), 'timeCorrection_'+name_1, filepath, 'sampling_rate')
		else:
			raise Exception('Error in naming convention of files or Error in storesList file')

	print("Timestamps corrected and converted to seconds.")
	#return timeRecStart, correctionIndex, timestampNew


# function to apply correction to control, signal and event timestamps 
def applyCorrection(filepath, timeForLightsTurnOn, event, displayName, naming):

	timeRecStart = read_hdf5('timeCorrection_'+naming, filepath, 'timeRecStart')[0]
	timestampNew = read_hdf5('timeCorrection_'+naming, filepath, 'timestampNew')
	correctionIndex = read_hdf5('timeCorrection_'+naming, filepath, 'correctionIndex')

	
	if 'control' in displayName.lower() or 'signal' in displayName.lower():
		arr = read_hdf5(event, filepath, 'data')
		if (arr==0).all()==True:
			arr = arr
		else:
			arr = arr[correctionIndex]
		write_hdf5(arr, displayName, filepath, 'data')
	else:
		arr = read_hdf5(event, filepath, 'timestamps')
		res = (arr>=timeRecStart).all()
		if res:
			arr = np.subtract(arr, timeRecStart)
			arr = np.subtract(arr, timeForLightsTurnOn)
		else:
			arr = np.subtract(arr, timeForLightsTurnOn)
		write_hdf5(arr, displayName+'_'+naming, filepath, 'ts')
		


# function to check if naming convention was followed while saving storeslist file
# and apply timestamps correction using the function applyCorrection
def decide_naming_convention_and_applyCorrection(filepath, timeForLightsTurnOn, event, displayName, storesList):

	print("Applying correction of timestamps to the data and event timestamps...")
	storesList = storesList[1,:]
	
	arr = []
	for i in range(storesList.shape[0]):
	    if 'control' in storesList[i].lower() or 'signal' in storesList[i].lower():
	        arr.append(storesList[i])

	arr = sorted(arr)
	arr = np.asarray(arr).reshape(2,-1)

	for i in range(arr.shape[1]):
		name_1 = arr[0,i].split('_')[-1]
		name_2 = arr[1,i].split('_')[-1]
		#dirname = os.path.dirname(path[i])
		if name_1==name_2:
			applyCorrection(filepath, timeForLightsTurnOn, event, displayName, name_1)
		else:
			raise Exception('Error in naming convention of files or Error in storesList file')

	print("Timestamps corrections applied to the data and event timestamps.")



# functino to plot z_score
def visualize_z_score(filepath):

	name = os.path.basename(filepath)

	path = glob.glob(os.path.join(filepath, 'z_score_*'))
	
	path = sorted(path)

	for i in range(len(path)):
		basename = (os.path.basename(path[i])).split('.')[0]
		name_1 = basename.split('_')[-1]
		x = read_hdf5('timeCorrection_'+name_1, filepath, 'timestampNew')
		y = read_hdf5('', path[i], 'data')
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(x,y)
		ax.set_title(basename)
		fig.suptitle(name)
	plt.show()

# function to plot deltaF/F
def visualize_dff(filepath):
	name = os.path.basename(filepath)

	path = glob.glob(os.path.join(filepath, 'dff_*'))
	
	path = sorted(path)

	for i in range(len(path)):
		basename = (os.path.basename(path[i])).split('.')[0]
		name_1 = basename.split('_')[-1]
		x = read_hdf5('timeCorrection_'+name_1, filepath, 'timestampNew')
		y = read_hdf5('', path[i], 'data')
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(x,y)
		ax.set_title(basename)
		fig.suptitle(name)
	plt.show()



def visualize(filepath, x, y1, y2, plot_name, removeArtifacts):
    

	# plotting control and signal data

	if (y1==0).all()==True:
		y1 = np.zeros(x.shape[0])

	name = os.path.basename(filepath)
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	line1, = ax1.plot(x,y1)
	ax1.set_title(plot_name[0])
	ax2 = fig.add_subplot(212)
	line2, = ax2.plot(x,y2)
	ax2.set_title(plot_name[1])
	fig.suptitle(name)

	hfont = {'fontname':'Helvetica'}

	if removeArtifacts==True:
		ax2.set_xlabel('Time(s) \n Note : Artifacts have been removed, but are not reflected in this plot.', **hfont)
	else:
		ax2.set_xlabel('Time(s)', **hfont)

	global coords
	coords = []

	# clicking any key on keyboard will draw a line on the plot so that user can see what chunks are selected
	def onclick(event):
		global ix, iy
		ix, iy = event.xdata, event.ydata
		print('x = %d, y = %d'%(
		    ix, iy))

		y1_max, y1_min = np.amax(y1), np.amin(y1)
		y2_max, y2_min = np.amax(y2), np.amin(y2)

		ax1.plot([ix,ix], [y1_max, y1_min], 'k--')
		ax2.plot([ix,ix], [y2_max, y2_min], 'k--')

		fig.canvas.draw()

		global coords
		coords.append((ix, iy))

		#if len(coords) == 2:
		#    fig.canvas.mpl_disconnect(cid)

		return coords

	# close the plot will save coordinates for all the selected chunks in the data
	def plt_close_event(event):
		global coords
		if coords and len(coords)>0:
			name_1 = plot_name[0].split('_')[-1]
			np.save(os.path.join(filepath, 'coordsForPreProcessing_'+name_1+'.npy'), coords)
			print('Coordinates file saved at {}'.format(os.path.join(filepath, 'coordsForPreProcessing_'+name_1+'.npy')))
		fig.canvas.mpl_disconnect(cid)
		coords = []


	cid = fig.canvas.mpl_connect('key_press_event', onclick)
	cid = fig.canvas.mpl_connect('close_event', plt_close_event)
	#multi = MultiCursor(fig.canvas, (ax1, ax2), color='g', lw=1, horizOn=False, vertOn=True)

	plt.show()
	#return fig

# function to plot control and signal, also provide a feature to select chunks for artifacts removal
def visualizeControlAndSignal(filepath, removeArtifacts):
	path_1 = find_files(filepath, 'control*', ignore_case=True) #glob.glob(os.path.join(filepath, 'control*'))
	
	path_2 = find_files(filepath, 'signal*', ignore_case=True) #glob.glob(os.path.join(filepath, 'signal*'))
	

	path = sorted(path_1 + path_2)
	
	if len(path)%2 != 0:
		raise Exception('There are not equal number of Control and Signal data')
	
	path = np.asarray(path).reshape(2,-1)
	
	for i in range(path.shape[1]):
		
		name_1 = ((os.path.basename(path[0,i])).split('.')[0]).split('_')
		name_2 = ((os.path.basename(path[1,i])).split('.')[0]).split('_')
		
		ts_path = os.path.join(filepath, 'timeCorrection_'+name_1[-1]+'.hdf5')
		ts = read_hdf5('', ts_path, 'timestampNew')
		
		control = read_hdf5('', path[0,i], 'data').reshape(-1)
		signal = read_hdf5('', path[1,i], 'data').reshape(-1)
			
		plot_name = [(os.path.basename(path[0,i])).split('.')[0], (os.path.basename(path[1,i])).split('.')[0]]
		visualize(filepath, ts, control, signal, plot_name, removeArtifacts)
		

# functino to check if the naming convention for saving storeslist file was followed or not
def decide_naming_convention(filepath):
	path_1 = find_files(filepath, 'control*', ignore_case=True) #glob.glob(os.path.join(filepath, 'control*'))
	
	path_2 = find_files(filepath, 'signal*', ignore_case=True) #glob.glob(os.path.join(filepath, 'signal*'))
	
	path = sorted(path_1 + path_2)
	if len(path)%2 != 0:
		raise Exception('There are not equal number of Control and Signal data')
	
	path = np.asarray(path).reshape(2,-1)

	return path


# function to read coordinates file which was saved by selecting chunks for artifacts removal
def fetchCoords(filepath, naming, data):

	path = os.path.join(filepath, 'coordsForPreProcessing_'+naming+'.npy')

	if not os.path.exists(path):
		coords = np.array([0, data[-1]])
	else:
		coords = np.load(os.path.join(filepath, 'coordsForPreProcessing_'+naming+'.npy'))[:,0]

	if coords.shape[0] % 2 != 0:
		raise Exception('Number of values in coordsForPreProcessing file is not even.')

	coords = coords.reshape(-1,2)

	return coords


# helper function to process control and signal timestamps
def eliminateData(filepath, timeForLightsTurnOn, event, sampling_rate, naming):
	
	ts = read_hdf5('timeCorrection_'+naming, filepath, 'timestampNew')
	data = read_hdf5(event, filepath, 'data').reshape(-1)
	coords = fetchCoords(filepath, naming, ts)

	if (data==0).all()==True:
		data = np.zeros(ts.shape[0])

	arr = np.array([])
	ts_arr = np.array([])
	for i in range(coords.shape[0]):

		index = np.where((ts>coords[i,0]) & (ts<coords[i,1]))[0]
		
		if len(arr)==0:
			arr = np.concatenate((arr, data[index]))
			sub = ts[index][0]-timeForLightsTurnOn
			new_ts = ts[index]-sub
			ts_arr = np.concatenate((ts_arr, new_ts))
		else:
			temp = data[index]
			#new = temp + (arr[-1]-temp[0])
			temp_ts = ts[index]
			new_ts = temp_ts - (temp_ts[0]-ts_arr[-1])
			arr = np.concatenate((arr, temp))
			ts_arr = np.concatenate((ts_arr, new_ts+(1/sampling_rate)))
	
	#print(arr.shape, ts_arr.shape)
	return arr, ts_arr


# helper function to align event timestamps with the control and signal timestamps
def eliminateTs(filepath, timeForLightsTurnOn, event, sampling_rate, naming):
	
	tsNew = read_hdf5('timeCorrection_'+naming, filepath, 'timestampNew')
	ts = read_hdf5(event+'_'+naming, filepath, 'ts').reshape(-1)
	coords = fetchCoords(filepath, naming, tsNew)

	ts_arr = np.array([])
	tsNew_arr = np.array([])
	for i in range(coords.shape[0]):
		tsNew_index = np.where((tsNew>coords[i,0]) & (tsNew<coords[i,1]))[0]
		ts_index = np.where((ts>coords[i,0]) & (ts<coords[i,1]))[0]
		
		if len(tsNew_arr)==0:
			sub = tsNew[tsNew_index][0]-timeForLightsTurnOn
			tsNew_arr = np.concatenate((tsNew_arr, tsNew[tsNew_index]-sub))
			ts_arr = np.concatenate((ts_arr, ts[ts_index]-sub))
		else:
			temp_tsNew = tsNew[tsNew_index]
			temp_ts = ts[ts_index]
			new_ts = temp_ts - (temp_tsNew[0]-tsNew_arr[-1])
			new_tsNew = temp_tsNew - (temp_tsNew[0]-tsNew_arr[-1])
			tsNew_arr = np.concatenate((tsNew_arr, new_tsNew+(1/sampling_rate)))
			ts_arr = np.concatenate((ts_arr, new_ts+(1/sampling_rate)))
	
	return ts_arr

# main function to align timestamps for control, signal and event timestamps for artifacts removal
def processTimestampsForArtifacts(filepath, timeForLightsTurnOn, events):

	print("Processing timestamps to get rid of artifacts...")
	storesList = events[1,:]
	
	path = decide_naming_convention(filepath)

	timestamp_dict = dict()
	for j in range(path.shape[1]):
		name_1 = ((os.path.basename(path[0,j])).split('.')[0]).split('_')
		name_2 = ((os.path.basename(path[1,j])).split('.')[0]).split('_')
		#dirname = os.path.dirname(path[i])
		if name_1[-1]==name_2[-1]:
			name = name_1[-1]
			sampling_rate = read_hdf5('timeCorrection_'+name, filepath, 'sampling_rate')[0]

			for i in range(len(storesList)):
				if 'control_'+name.lower() in storesList[i].lower() or 'signal_'+name.lower() in storesList[i].lower():       # changes done
					data, timestampNew = eliminateData(filepath, timeForLightsTurnOn, storesList[i], sampling_rate, name)
					write_hdf5(data, storesList[i], filepath, 'data')
				else:
					if 'control' in storesList[i].lower() or 'signal' in storesList[i].lower():
						continue
					else:
						ts = eliminateTs(filepath, timeForLightsTurnOn, storesList[i], sampling_rate, name)
						write_hdf5(ts, storesList[i]+'_'+name, filepath, 'ts')


			#timestamp_dict[name] = timestampNew
			write_hdf5(timestampNew, 'timeCorrection_'+name, filepath, 'timestampNew')
		else:
			raise Exception('Error in naming convention of files or Error in storesList file')

	print("Timestamps processed and artifacts are removed.")


# function to compute deltaF/F using fitted control channel and filtered signal channel
def deltaFF(signal, control):
    
	res = np.subtract(signal, control)
	normData = np.divide(res, control)
	#deltaFF = normData
	normData = normData*100

	return normData

# function to fit control channel to signal channel
def controlFit(control, signal):
    
	p = np.polyfit(control, signal, 1)
	arr = (p[0]*control)+p[1]
	return arr


# function to filter control and signal channel, also execute above two function : controlFit and deltaFF
# function will also take care if there is only signal channel and no control channel
# if there is only signal channel, z-score will be computed using just signal channel
def execute_controlFit_dff(control, signal):

	b = np.divide(np.ones((100,)), 100)
	a = 1

	if (control==0).all()==True:
		signal_smooth = ss.filtfilt(b, a, signal)
		return signal_smooth
	else:
		control_smooth = ss.filtfilt(b, a, control)
		signal_smooth = ss.filtfilt(b, a, signal)
		control_fit = controlFit(control_smooth, signal_smooth)
		norm_data = deltaFF(signal_smooth, control_fit)
		return norm_data


# helper function to compute z-score and deltaF/F

def helper_z_score(control, signal, filepath, name, inputParameters):     #helper_z_score(control_smooth, signal_smooth):

	removeArtifacts = inputParameters['removeArtifacts']
	tsNew = read_hdf5('timeCorrection_'+name, filepath, 'timestampNew')
	coords_path = os.path.join(filepath, 'coordsForPreProcessing_'+name+'.npy')

	print("Remove Artifacts : ", removeArtifacts)

	if (control==0).all()==True:
		control = np.zeros(tsNew.shape[0])
	
	z_score_arr, norm_data_arr = np.array([]), np.array([])

	if removeArtifacts==True:
		coords = fetchCoords(filepath, name, tsNew)

		# for artifacts removal, each chunk which was selected by user is being processed individually and then 
		# z-score is calculated
		for i in range(coords.shape[0]):
			tsNew_index = np.where((tsNew>coords[i,0]) & (tsNew<coords[i,1]))[0]
			control_arr = control[tsNew_index]
			signal_arr = signal[tsNew_index]
			#print(coords[i,:], control_arr.shape, signal_arr.shape)
			#control_smooth = ss.filtfilt(b, a, control_arr)
			#signal_smooth = ss.filtfilt(b, a, signal_arr)
			#control_fit = controlFit(control_smooth, signal_smooth)
			norm_data = execute_controlFit_dff(control_arr, signal_arr)
			norm_data_arr = np.concatenate((norm_data_arr, norm_data))

		res = np.subtract(norm_data_arr, np.nanmean(norm_data_arr))
		z_score = np.divide(res, np.nanstd(norm_data_arr))
		z_score_arr = np.concatenate((z_score_arr, z_score))
			
	else:
		#control_smooth = ss.filtfilt(b, a, control)
		#signal_smooth = ss.filtfilt(b, a, signal)
		#control_fit = controlFit(control_smooth, signal_smooth)
		norm_data = execute_controlFit_dff(control, signal)
		res = np.subtract(norm_data, np.nanmean(norm_data))
		z_score = np.divide(res, np.nanstd(norm_data))
		z_score_arr = np.concatenate((z_score_arr, z_score))
		norm_data_arr = np.concatenate((norm_data_arr, norm_data))

	return z_score_arr, norm_data_arr


# compute z-score and deltaF/F and save it to hdf5 file
def compute_z_score(filepath, inputParameters):

	print("Computing z-score for each of the data...")
	remove_artifacts = inputParameters['removeArtifacts']


	path_1 = find_files(filepath, 'control*', ignore_case=True) #glob.glob(os.path.join(filepath, 'control*'))
	path_2 = find_files(filepath, 'signal*', ignore_case=True) #glob.glob(os.path.join(filepath, 'signal*'))
	
	path = sorted(path_1 + path_2)


	b = np.divide(np.ones((100,)), 100)
	a = 1

	if len(path)%2 != 0:
		raise Exception('There are not equal number of Control and Signal data')

	path = np.asarray(path).reshape(2,-1)

	for i in range(path.shape[1]):
		name_1 = ((os.path.basename(path[0,i])).split('.')[0]).split('_')
		name_2 = ((os.path.basename(path[1,i])).split('.')[0]).split('_')
		#dirname = os.path.dirname(path[i])
		
		if name_1[-1]==name_2[-1]:
			name = name_1[-1]
			control = read_hdf5('', path[0,i], 'data').reshape(-1)
			signal = read_hdf5('', path[1,i], 'data').reshape(-1)
			#control_smooth = ss.filtfilt(b, a, control)
			#signal_smooth = ss.filtfilt(b, a, signal)
			#_score, dff = helper_z_score(control_smooth, signal_smooth)
			z_score, dff = helper_z_score(control, signal, filepath, name, inputParameters)
			if remove_artifacts==True:
				write_hdf5(z_score, 'z_score_'+name, filepath, 'data')
				write_hdf5(dff, 'dff_'+name, filepath, 'data')
			else:
				write_hdf5(z_score, 'z_score_'+name, filepath, 'data')
				write_hdf5(dff, 'dff_'+name, filepath, 'data')
		else:
			raise Exception('Error in naming convention of files or Error in storesList file')


	print("z-score for the data computed.")
	


# function to execute timestamps corrections using functions timestampCorrection and decide_naming_convention_and_applyCorrection
def execute_timestamp_correction(folderNames, timeForLightsTurnOn):

	for i in range(len(folderNames)):
		filepath = folderNames[i]
		storesListPath = glob.glob(os.path.join(filepath, '*_output_*'))
		for j in range(len(storesListPath)):
			filepath = storesListPath[j]
			storesList = np.genfromtxt(os.path.join(filepath, 'storesList.csv'), dtype='str', delimiter=',')

			timestampCorrection(filepath, timeForLightsTurnOn, storesList)


			for k in range(storesList.shape[1]):
				decide_naming_convention_and_applyCorrection(filepath, timeForLightsTurnOn, storesList[0,k], storesList[1,k], storesList)


# for combining data, reading storeslist file from both data and create a new storeslist array
def check_storeslistfile(folderNames):
	storesList = np.array([[],[]])
	for i in range(len(folderNames)):
		filepath = folderNames[i]
		storesListPath = glob.glob(os.path.join(filepath, '*_output_*'))
		for j in range(len(storesListPath)):
			filepath = storesListPath[j]
			storesList = np.concatenate((storesList, np.genfromtxt(os.path.join(filepath, 'storesList.csv'), dtype='str', delimiter=',')), axis=1)

	storesList = np.unique(storesList, axis=1)
	
	return storesList

def get_all_stores_for_combining_data(folderNames):
	op = []
	for i in range(100):
		temp = []
		match = r'[\s\S]*'+'_output_'+str(i)
		for j in folderNames:
			temp.append(re.findall(match, j))
		temp = sorted(list(np.concatenate(temp).flatten()))
		if len(temp)>0:
			op.append(temp)

	return op


# function to combine data when there are two different data files for the same recording session
# it will combine the data, do timestamps processing and save the combined data in the first output folder.
def combineData(folderNames, timeForLightsTurnOn, storesList):

	print("Combining Data from different data files...")
	op_folder = []
	for i in range(len(folderNames)):
		filepath = folderNames[i]
		op_folder.append(glob.glob(os.path.join(filepath, '*_output_*')))


	op_folder = list(np.concatenate(op_folder).flatten())
	sampling_rate_fp = []
	for i in range(len(folderNames)):
		filepath = folderNames[i]
		storesListPath = glob.glob(os.path.join(filepath, '*_output_*'))
		for j in range(len(storesListPath)):
			filepath = storesListPath[j]
			storesList_new = np.genfromtxt(os.path.join(filepath, 'storesList.csv'), dtype='str', delimiter=',')
			sampling_rate_fp.append(glob.glob(os.path.join(filepath, 'timeCorrection_*')))

	# check if sampling rate is same for both data
	sampling_rate_fp = np.concatenate(sampling_rate_fp)
	sampling_rate = []
	for i in range(sampling_rate_fp.shape[0]):
		sampling_rate.append(read_hdf5('', sampling_rate_fp[i], 'sampling_rate'))

	res = all(i == sampling_rate[0] for i in sampling_rate)
	if res==False:
		raise Exception('To combine the data, sampling rate for both the data should be same.')
	
	# get the output folders informatinos
	op = get_all_stores_for_combining_data(op_folder)
	
	# processing timestamps for combining the data
	processTimestampsForCombiningData(op, timeForLightsTurnOn, storesList, sampling_rate[0])
	
	print("Data is combined from different data files.")

	return op


# function to compute z-score and deltaF/F using functions : compute_z_score and/or processTimestampsForArtifacts
def execute_zscore(folderNames, inputParameters, timeForLightsTurnOn, remove_artifacts, plot_zScore_dff, combine_data):

	storesListPath = []
	for i in range(len(folderNames)):
		if combine_data==True:
			storesListPath.append([folderNames[i][0]])
		else:
			filepath = folderNames[i]
			storesListPath.append(glob.glob(os.path.join(filepath, '*_output_*')))
	
	storesListPath = np.concatenate(storesListPath)
	
	for j in range(len(storesListPath)):
		filepath = storesListPath[j]
		storesList = np.genfromtxt(os.path.join(filepath, 'storesList.csv'), dtype='str', delimiter=',')

		if remove_artifacts==True:
			print("Removing Artifacts from the data and correcting timestamps...")
			compute_z_score(filepath, inputParameters)
			processTimestampsForArtifacts(filepath, timeForLightsTurnOn, storesList)
			visualizeControlAndSignal(filepath, remove_artifacts)
			print("Artifacts from the data are removed and timestamps are corrected.")
		else:
			compute_z_score(filepath, inputParameters)
			visualizeControlAndSignal(filepath, remove_artifacts)

		if plot_zScore_dff=='z_score':
			visualize_z_score(filepath)
		if plot_zScore_dff=='dff':
			visualize_dff(filepath)
		if plot_zScore_dff=='Both':
			visualize_z_score(filepath)
			visualize_dff(filepath)

	#plt.show()
	print("Signal data and event timestamps are extracted.")


def extractTsAndSignal(inputParametersPath):

	print("Extracting signal data and event timestamps...")

	with open(inputParametersPath) as f:	
		inputParameters = json.load(f)

	#storesList = np.genfromtxt(inputParameters['storesListPath'], dtype='str', delimiter=',')

	folderNames = inputParameters['folderNames']
	timeForLightsTurnOn = inputParameters['timeForLightsTurnOn']
	remove_artifacts = inputParameters['removeArtifacts']
	plot_zScore_dff = inputParameters['plot_zScore_dff']
	combine_data = inputParameters['combine_data']

	print("Remove Artifacts : ", remove_artifacts)
	print("Combine Data : ", combine_data)
	#print(type(remove_artifacts))

	if combine_data==False:
		execute_timestamp_correction(folderNames, timeForLightsTurnOn)
		execute_zscore(folderNames, inputParameters, timeForLightsTurnOn, remove_artifacts, plot_zScore_dff, combine_data)
	else:
		execute_timestamp_correction(folderNames, timeForLightsTurnOn)
		storesList = check_storeslistfile(folderNames)
		op_folder = combineData(folderNames, timeForLightsTurnOn, storesList)
		execute_zscore(op_folder, inputParameters, timeForLightsTurnOn, remove_artifacts, plot_zScore_dff, combine_data)
		

	




			