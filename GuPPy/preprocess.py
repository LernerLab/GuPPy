import os
import sys
import json
import glob
import time
import re
import fnmatch
import numpy as np 
import h5py
import math
import shutil
from scipy import signal as ss
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
from combineDataFn import processTimestampsForCombiningData
plt.switch_backend('TKAgg')


# find files by ignoring the case sensitivity
def find_files(path, glob_path, ignore_case = False):
	rule = re.compile(fnmatch.translate(glob_path), re.IGNORECASE) if ignore_case \
			else re.compile(fnmatch.translate(glob_path))

	no_bytes_path = os.listdir(os.path.expanduser(path))
	str_path = []

	# converting byte object to string
	for x in no_bytes_path:
		try:
			str_path.append(x.decode('utf-8'))
		except:
			str_path.append(x)
	return [os.path.join(path,n) for n in str_path if rule.match(n)]


# curve fit exponential function
def curveFitFn(x,a,b,c):
    return a+(b*np.exp(-(1/c)*x))


# helper function to create control channel using signal channel
# by curve fitting signal channel to exponential function
# when there is no isosbestic control channel is present
def helper_create_control_channel(signal, timestamps, window):
	# check if window is greater than signal shape
	if window>signal.shape[0]:
		window = ((window+1)/2)+1

	filtered_signal = ss.savgol_filter(signal, window_length=window, polyorder=3)

	p0 = [5,50,60]

	try:
		popt, pcov = curve_fit(curveFitFn, timestamps, filtered_signal, p0)
	except Exception as e:
		print(e)

	#print('Curve Fit Parameters : ', popt)
	control = curveFitFn(timestamps,*popt)

	return control

# main function to create control channel using
# signal channel and save it to a file
def create_control_channel(filepath, arr, window=5001):

	storenames = arr[0,:]
	storesList = arr[1,:]

	
	for i in range(storesList.shape[0]):
		event_name, event = storesList[i], storenames[i]
		if 'control' in event_name.lower() and 'cntrl' in event.lower():
			print('Creating control channel from signal channel using curve-fitting')
			name = event_name.split('_')[-1]
			signal = read_hdf5('signal_'+name, filepath, 'data')
			timestampNew = read_hdf5('timeCorrection_'+name, filepath, 'timestampNew')

			control = helper_create_control_channel(signal, timestampNew, window)

			write_hdf5(control, event_name, filepath, 'data')

			print('Control channel from signal channel created using curve-fitting')


# function to add control channel when there is no 
# isosbestic control channel and update the storeslist file
def add_control_channel(filepath, arr):

	storenames = arr[0,:]
	storesList = np.char.lower(arr[1,:])

	keep_control = np.array([])
	# check a case if there is isosbestic control channel present
	for i in range(storesList.shape[0]):
		if 'control' in storesList[i].lower():
			name = storesList[i].split('_')[-1]
			new_str = 'signal_'+str(name).lower()
			find_signal = [True for i in storesList if i==new_str]
			if len(find_signal)>1:
				raise Exception('Error in naming convention of files or Error in storesList file')
			if len(find_signal)==0:
				raise Exception("Isosbectic control channel parameter is set to False and still \
							 	 storeslist file shows there is control channel present")
		else:
			continue
	
	for i in range(storesList.shape[0]):
		if 'signal' in storesList[i].lower():
			name = storesList[i].split('_')[-1]
			new_str = 'control_'+str(name).lower()
			find_signal = [True for i in storesList if i==new_str]
			if len(find_signal)==0:
				src, dst = os.path.join(filepath, arr[0,i]+'.hdf5'), os.path.join(filepath, 'cntrl'+str(i)+'.hdf5')
				shutil.copyfile(src,dst)
				arr = np.concatenate((arr, [['cntrl'+str(i)],['control_'+str(arr[1,i].split('_')[-1])]]), axis=1)

	np.savetxt(os.path.join(filepath, 'storesList.csv'), arr, delimiter=",", fmt='%s')

	return arr

# check if dealing with TDT files or csv files
def check_TDT(filepath):
	path = glob.glob(os.path.join(filepath, '*.tsq'))
	if len(path)>0:
		return True
	else:
		return False

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


# function to check control and signal channel has same length
# if not, take a smaller length and do pre-processing
def check_cntrl_sig_length(filepath, channels_arr, storenames, storesList):

	indices = []
	for i in range(channels_arr.shape[1]):
		idx_c = np.where(storesList==channels_arr[0,i])[0]
		idx_s = np.where(storesList==channels_arr[1,i])[0]
		control = read_hdf5(storenames[idx_c[0]], filepath, 'data') 
		signal = read_hdf5(storenames[idx_s[0]], filepath, 'data')
		if control.shape[0]<signal.shape[0]:
			indices.append(storesList[idx_c[0]])
		elif control.shape[0]>signal.shape[0]:
			indices.append(storesList[idx_s[0]])
		else:
			indices.append(storesList[idx_s[0]])

	return indices


# function to correct timestamps after eliminating first few seconds of the data (for csv data)
def timestampCorrection_csv(filepath, timeForLightsTurnOn, storesList):
	
	print("Correcting timestamps by getting rid of the first {} seconds and convert timestamps to seconds...".format(timeForLightsTurnOn))
	storenames = storesList[0,:]
	storesList = storesList[1,:]

	arr = []
	for i in range(storesList.shape[0]):
		if 'control' in storesList[i].lower() or 'signal' in storesList[i].lower():
			arr.append(storesList[i])

	arr = sorted(arr, key=str.casefold)
	try:
		arr = np.asarray(arr).reshape(2,-1)
	except:
		raise Exception('Error in saving stores list file or spelling mistake for control or signal')

	indices = check_cntrl_sig_length(filepath, arr, storenames, storesList)

	for i in range(arr.shape[1]):
		name_1 = arr[0,i].split('_')[-1]
		name_2 = arr[1,i].split('_')[-1]
		#dirname = os.path.dirname(path[i])
		idx = np.where(storesList==indices[i])[0]

		if idx.shape[0]==0:
			raise Exception('{} does not exist in the stores list file.'.format(arr[0,i]))

		timestamp = read_hdf5(storenames[idx][0], filepath, 'timestamps')
		sampling_rate = read_hdf5(storenames[idx][0], filepath, 'sampling_rate')

		if name_1==name_2:
			correctionIndex = np.where(timestamp>=timeForLightsTurnOn)[0]
			timestampNew = timestamp[correctionIndex]
			write_hdf5(timestampNew, 'timeCorrection_'+name_1, filepath, 'timestampNew')
			write_hdf5(correctionIndex, 'timeCorrection_'+name_1, filepath, 'correctionIndex')
			write_hdf5(np.asarray(sampling_rate), 'timeCorrection_'+name_1, filepath, 'sampling_rate')

		else:
			raise Exception('Error in naming convention of files or Error in storesList file')

	print("Timestamps corrected and converted to seconds.")



# function to correct timestamps after eliminating first few seconds of the data (for TDT data)
def timestampCorrection_tdt(filepath, timeForLightsTurnOn, storesList):

	print("Correcting timestamps by getting rid of the first {} seconds and convert timestamps to seconds...".format(timeForLightsTurnOn))
	storenames = storesList[0,:]
	storesList = storesList[1,:]

	arr = []
	for i in range(storesList.shape[0]):
		if 'control' in storesList[i].lower() or 'signal' in storesList[i].lower():
			arr.append(storesList[i])

	arr = sorted(arr, key=str.casefold)

	try:
		arr = np.asarray(arr).reshape(2,-1)
	except:
		raise Exception('Error in saving stores list file or spelling mistake for control or signal')

	indices = check_cntrl_sig_length(filepath, arr, storenames, storesList)

	for i in range(arr.shape[1]):
		name_1 = arr[0,i].split('_')[-1]
		name_2 = arr[1,i].split('_')[-1]
		#dirname = os.path.dirname(path[i])
		idx = np.where(storesList==indices[i])[0]

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

	cond = check_TDT(os.path.dirname(filepath))

	if cond==True:
		timeRecStart = read_hdf5('timeCorrection_'+naming, filepath, 'timeRecStart')[0]
	
	timestampNew = read_hdf5('timeCorrection_'+naming, filepath, 'timestampNew')
	correctionIndex = read_hdf5('timeCorrection_'+naming, filepath, 'correctionIndex')

	if 'control' in displayName.lower() or 'signal' in displayName.lower():
		split_name = displayName.split('_')[-1]
		if split_name==naming:
			pass 
		else:
			correctionIndex = read_hdf5('timeCorrection_'+split_name, filepath, 'correctionIndex')
		arr = read_hdf5(event, filepath, 'data')
		if (arr==0).all()==True:
			arr = arr
		else:
			arr = arr[correctionIndex]
		write_hdf5(arr, displayName, filepath, 'data')
	else:
		arr = read_hdf5(event, filepath, 'timestamps')
		if cond==True:
			res = (arr>=timeRecStart).all()
			if res==True:
				arr = np.subtract(arr, timeRecStart)
				arr = np.subtract(arr, timeForLightsTurnOn)
			else:
				arr = np.subtract(arr, timeForLightsTurnOn)
		else:
			arr = np.subtract(arr, timeForLightsTurnOn)
		write_hdf5(arr, displayName+'_'+naming, filepath, 'ts')
		
	#if isosbestic_control==False and 'control' in displayName.lower():
	#	control = create_control_channel(filepath, displayName)
	#	write_hdf5(control, displayName, filepath, 'data')


# function to check if naming convention was followed while saving storeslist file
# and apply timestamps correction using the function applyCorrection
def decide_naming_convention_and_applyCorrection(filepath, timeForLightsTurnOn, event, displayName, storesList):

	print("Applying correction of timestamps to the data and event timestamps...")
	storesList = storesList[1,:]

	arr = []
	for i in range(storesList.shape[0]):
		if 'control' in storesList[i].lower() or 'signal' in storesList[i].lower():
			arr.append(storesList[i])

	arr = sorted(arr, key=str.casefold)
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
	#plt.show()

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
	#plt.show()



def visualize(filepath, x, y1, y2, y3, plot_name, removeArtifacts):
    

	# plotting control and signal data

	if (y1==0).all()==True:
		y1 = np.zeros(x.shape[0])

	coords_path = os.path.join(filepath, 'coordsForPreProcessing_'+plot_name[0].split('_')[-1]+'.npy')
	name = os.path.basename(filepath)
	fig = plt.figure()
	ax1 = fig.add_subplot(311)
	line1, = ax1.plot(x,y1)
	ax1.set_title(plot_name[0])
	ax2 = fig.add_subplot(312)
	line2, = ax2.plot(x,y2)
	ax2.set_title(plot_name[1])
	ax3 = fig.add_subplot(313)
	line3, = ax3.plot(x,y2)
	line3, = ax3.plot(x,y3)
	ax3.set_title(plot_name[2])
	fig.suptitle(name)

	hfont = {'fontname':'Helvetica'}

	if removeArtifacts==True and os.path.exists(coords_path):
		ax3.set_xlabel('Time(s) \n Note : Artifacts have been removed, but are not reflected in this plot.', **hfont)
	else:
		ax3.set_xlabel('Time(s)', **hfont)

	global coords
	coords = []

	# clicking 'space' key on keyboard will draw a line on the plot so that user can see what chunks are selected
	# and clicking 'd' key on keyboard will deselect the selected point
	def onclick(event):
		#global ix, iy

		if event.key == ' ':
			ix, iy = event.xdata, event.ydata
			print('x = %d, y = %d'%(
			    ix, iy))

			y1_max, y1_min = np.amax(y1), np.amin(y1)
			y2_max, y2_min = np.amax(y2), np.amin(y2)

			#ax1.plot([ix,ix], [y1_max, y1_min], 'k--')
			#ax2.plot([ix,ix], [y2_max, y2_min], 'k--')

			ax1.axvline(ix, c='black', ls='--')
			ax2.axvline(ix, c='black', ls='--')
			ax3.axvline(ix, c='black', ls='--')

			fig.canvas.draw()

			global coords
			coords.append((ix, iy))

			#if len(coords) == 2:
			#    fig.canvas.mpl_disconnect(cid)

			return coords

		elif event.key == 'd':
			if len(coords)>0:
				print('x = %d, y = %d; deleted'%(
			    	coords[-1][0], coords[-1][1]))
				del coords[-1]
				ax1.lines[-1].remove()
				ax2.lines[-1].remove()
				ax3.lines[-1].remove()
				fig.canvas.draw()

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

	#plt.show()
	#return fig

# function to plot control and signal, also provide a feature to select chunks for artifacts removal
def visualizeControlAndSignal(filepath, removeArtifacts):
	path_1 = find_files(filepath, 'control_*', ignore_case=True) #glob.glob(os.path.join(filepath, 'control*'))
	
	path_2 = find_files(filepath, 'signal_*', ignore_case=True) #glob.glob(os.path.join(filepath, 'signal*'))
	

	path = sorted(path_1 + path_2, key=str.casefold)
	
	if len(path)%2 != 0:
		raise Exception('There are not equal number of Control and Signal data')
	
	path = np.asarray(path).reshape(2,-1)
	
	for i in range(path.shape[1]):
		
		name_1 = ((os.path.basename(path[0,i])).split('.')[0]).split('_')
		name_2 = ((os.path.basename(path[1,i])).split('.')[0]).split('_')
		
		ts_path = os.path.join(filepath, 'timeCorrection_'+name_1[-1]+'.hdf5')
		cntrl_sig_fit_path = os.path.join(filepath, 'cntrl_sig_fit_'+name_1[-1]+'.hdf5')
		ts = read_hdf5('', ts_path, 'timestampNew')
		
		control = read_hdf5('', path[0,i], 'data').reshape(-1)
		signal = read_hdf5('', path[1,i], 'data').reshape(-1)
		cntrl_sig_fit = read_hdf5('', cntrl_sig_fit_path, 'data').reshape(-1)

		plot_name = [(os.path.basename(path[0,i])).split('.')[0], 
					 (os.path.basename(path[1,i])).split('.')[0],
					 (os.path.basename(cntrl_sig_fit_path)).split('.')[0]]
		visualize(filepath, ts, control, signal, cntrl_sig_fit, plot_name, removeArtifacts)
		

# functino to check if the naming convention for saving storeslist file was followed or not
def decide_naming_convention(filepath):
	path_1 = find_files(filepath, 'control_*', ignore_case=True) #glob.glob(os.path.join(filepath, 'control*'))
	
	path_2 = find_files(filepath, 'signal_*', ignore_case=True) #glob.glob(os.path.join(filepath, 'signal*'))
	
	path = sorted(path_1 + path_2, key=str.casefold)
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

# adding nan values to removed chunks 
# when using artifacts removal method - replace with NaN
def addingNaNValues(filepath, event, naming):
	
	ts = read_hdf5('timeCorrection_'+naming, filepath, 'timestampNew')
	data = read_hdf5(event, filepath, 'data').reshape(-1)
	coords = fetchCoords(filepath, naming, ts)

	if (data==0).all()==True:
		data = np.zeros(ts.shape[0])

	arr = np.array([])
	ts_index = np.arange(ts.shape[0])
	for i in range(coords.shape[0]):

		index = np.where((ts>coords[i,0]) & (ts<coords[i,1]))[0]
		arr = np.concatenate((arr, index))
	
	nan_indices = list(set(ts_index).symmetric_difference(arr))
	data[nan_indices] = np.nan

	return data

# remove event TTLs which falls in the removed chunks
# when using artifacts removal method - replace with NaN
def removeTTLs(filepath, event, naming):
	tsNew = read_hdf5('timeCorrection_'+naming, filepath, 'timestampNew')
	ts = read_hdf5(event+'_'+naming, filepath, 'ts').reshape(-1)
	coords = fetchCoords(filepath, naming, tsNew)

	ts_arr = np.array([])
	for i in range(coords.shape[0]):
		ts_index = np.where((ts>coords[i,0]) & (ts<coords[i,1]))[0]
		ts_arr = np.concatenate((ts_arr, ts[ts_index]))
	
	return ts_arr

def addingNaNtoChunksWithArtifacts(filepath, events):
	print("Replacing chunks with artifacts by NaN values.")
	storesList = events[1,:]

	path = decide_naming_convention(filepath)

	for j in range(path.shape[1]):
		name_1 = ((os.path.basename(path[0,j])).split('.')[0]).split('_')
		name_2 = ((os.path.basename(path[1,j])).split('.')[0]).split('_')
		#dirname = os.path.dirname(path[i])
		if name_1[-1]==name_2[-1]:
			name = name_1[-1]
			sampling_rate = read_hdf5('timeCorrection_'+name, filepath, 'sampling_rate')[0]
			for i in range(len(storesList)):
				if 'control_'+name.lower() in storesList[i].lower() or 'signal_'+name.lower() in storesList[i].lower():       # changes done
					data = addingNaNValues(filepath, storesList[i], name)
					write_hdf5(data, storesList[i], filepath, 'data')
				else:
					if 'control' in storesList[i].lower() or 'signal' in storesList[i].lower():
						continue
					else:
						ts = removeTTLs(filepath, storesList[i], name)
						write_hdf5(ts, storesList[i]+'_'+name, filepath, 'ts')
				
		else:
			raise Exception('Error in naming convention of files or Error in storesList file')

# main function to align timestamps for control, signal and event timestamps for artifacts removal
def processTimestampsForArtifacts(filepath, timeForLightsTurnOn, events):

	print("Processing timestamps to get rid of artifacts using concatenate method...")
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

	print("Timestamps processed, artifacts are removed and good chunks are concatenated.")


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
def execute_controlFit_dff(control, signal, isosbestic_control, filter_window):

	b = np.divide(np.ones((filter_window,)), filter_window)
	a = 1

	if isosbestic_control==False:
		signal_smooth = ss.filtfilt(b, a, signal)
		control_fit = controlFit(control, signal_smooth)
		norm_data = deltaFF(signal_smooth, control_fit)
	else:
		control_smooth = ss.filtfilt(b, a, control)
		signal_smooth = ss.filtfilt(b, a, signal)
		control_fit = controlFit(control_smooth, signal_smooth)
		norm_data = deltaFF(signal_smooth, control_fit)
	
	return norm_data, control_fit

# function to compute z-score based on z-score computation method
def z_score_computation(dff, timestamps, inputParameters):

	zscore_method = inputParameters['zscore_method']
	baseline_start, baseline_end = inputParameters['baselineWindowStart'], inputParameters['baselineWindowEnd']

	if zscore_method=='standard z-score':
		numerator = np.subtract(dff, np.nanmean(dff))
		zscore = np.divide(numerator, np.nanstd(dff))
	elif zscore_method=='baseline z-score':
		idx = np.where((timestamps>baseline_start) & (timestamps<baseline_end))[0]
		if idx.shape[0]==0:
			raise Exception('Baseline Window Parameters for baseline z-score computation zscore_method \
							are not correct.')
		else:
			baseline_mean = np.nanmean(dff[idx]) 
			baseline_std = np.nanstd(dff[idx])
			numerator = np.subtract(dff, baseline_mean)
			zscore = np.divide(numerator, baseline_std)
	else:
		median = np.median(dff)
		mad = np.median(np.abs(dff-median))
		numerator = 0.6745*(dff-median)
		zscore = np.divide(numerator, mad)

	return zscore

# helper function to compute z-score and deltaF/F
def helper_z_score(control, signal, filepath, name, inputParameters):     #helper_z_score(control_smooth, signal_smooth):

	removeArtifacts = inputParameters['removeArtifacts']
	artifactsRemovalMethod = inputParameters['artifactsRemovalMethod']
	filter_window = inputParameters['filter_window']

	isosbestic_control = inputParameters['isosbestic_control']
	tsNew = read_hdf5('timeCorrection_'+name, filepath, 'timestampNew')
	coords_path = os.path.join(filepath, 'coordsForPreProcessing_'+name+'.npy')

	print("Remove Artifacts : ", removeArtifacts)

	if (control==0).all()==True:
		control = np.zeros(tsNew.shape[0])
	
	z_score_arr = np.array([])
	norm_data_arr = np.full(tsNew.shape[0], np.nan)
	control_fit_arr = np.full(tsNew.shape[0], np.nan)
	temp_control_arr = np.full(tsNew.shape[0], np.nan)

	if removeArtifacts==True:
		coords = fetchCoords(filepath, name, tsNew)

		# for artifacts removal, each chunk which was selected by user is being processed individually and then 
		# z-score is calculated
		for i in range(coords.shape[0]):
			tsNew_index = np.where((tsNew>coords[i,0]) & (tsNew<coords[i,1]))[0]
			if isosbestic_control==False:
				control_arr = helper_create_control_channel(signal[tsNew_index], tsNew[tsNew_index], window=101)
				signal_arr = signal[tsNew_index]
				norm_data, control_fit = execute_controlFit_dff(control_arr, signal_arr, 
																isosbestic_control, filter_window)
				temp_control_arr[tsNew_index] = control_arr
				if i<coords.shape[0]-1:
					blank_index = np.where((tsNew>coords[i,1]) & (tsNew<coords[i+1,0]))[0]
					temp_control_arr[blank_index] = np.full(blank_index.shape[0], np.nan)
			else:
				control_arr = control[tsNew_index]
				signal_arr = signal[tsNew_index]
				norm_data, control_fit = execute_controlFit_dff(control_arr, signal_arr, 
					    									    isosbestic_control, filter_window)
			norm_data_arr[tsNew_index] = norm_data 
			control_fit_arr[tsNew_index] = control_fit 

		if artifactsRemovalMethod=='concatenate':
			norm_data_arr = norm_data_arr[~np.isnan(norm_data_arr)]
			control_fit_arr = control_fit_arr[~np.isnan(control_fit_arr)]
		z_score = z_score_computation(norm_data_arr, tsNew, inputParameters)
		z_score_arr = np.concatenate((z_score_arr, z_score))
	else:
		tsNew_index = np.arange(tsNew.shape[0])
		norm_data, control_fit = execute_controlFit_dff(control, signal, 
														isosbestic_control, filter_window)
		z_score = z_score_computation(norm_data, tsNew, inputParameters)
		z_score_arr = np.concatenate((z_score_arr, z_score))
		norm_data_arr[tsNew_index] = norm_data #np.concatenate((norm_data_arr, norm_data))
		control_fit_arr[tsNew_index] = control_fit #np.concatenate((control_fit_arr, control_fit))

	# handle the case if there are chunks being cut in the front and the end
	if isosbestic_control==False and removeArtifacts==True:
		coords = coords.flatten()
		# front chunk 
		idx = np.where((tsNew>=tsNew[0]) & (tsNew<coords[0]))[0]
		temp_control_arr[idx] = np.full(idx.shape[0], np.nan)
		# end chunk
		idx = np.where((tsNew>coords[-1]) & (tsNew<=tsNew[-1]))[0]
		temp_control_arr[idx] = np.full(idx.shape[0], np.nan)
		write_hdf5(temp_control_arr, 'control_'+name, filepath, 'data')

	return z_score_arr, norm_data_arr, control_fit_arr


# compute z-score and deltaF/F and save it to hdf5 file
def compute_z_score(filepath, inputParameters):

	print("Computing z-score for each of the data...")
	remove_artifacts = inputParameters['removeArtifacts']


	path_1 = find_files(filepath, 'control_*', ignore_case=True) #glob.glob(os.path.join(filepath, 'control*'))
	path_2 = find_files(filepath, 'signal_*', ignore_case=True) #glob.glob(os.path.join(filepath, 'signal*'))
	

	path = sorted(path_1 + path_2, key=str.casefold)


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
			z_score, dff, control_fit = helper_z_score(control, signal, filepath, name, inputParameters)
			if remove_artifacts==True:
				write_hdf5(z_score, 'z_score_'+name, filepath, 'data')
				write_hdf5(dff, 'dff_'+name, filepath, 'data')
				write_hdf5(control_fit, 'cntrl_sig_fit_'+name, filepath, 'data')
			else:
				write_hdf5(z_score, 'z_score_'+name, filepath, 'data')
				write_hdf5(dff, 'dff_'+name, filepath, 'data')
				write_hdf5(control_fit, 'cntrl_sig_fit_'+name, filepath, 'data')
		else:
			raise Exception('Error in naming convention of files or Error in storesList file')


	print("z-score for the data computed.")
	


# function to execute timestamps corrections using functions timestampCorrection and decide_naming_convention_and_applyCorrection
def execute_timestamp_correction(folderNames, timeForLightsTurnOn, isosbestic_control):


	for i in range(len(folderNames)):
		filepath = folderNames[i]
		storesListPath = glob.glob(os.path.join(filepath, '*_output_*'))
		cond = check_TDT(folderNames[i])

		for j in range(len(storesListPath)):
			filepath = storesListPath[j]
			storesList = np.genfromtxt(os.path.join(filepath, 'storesList.csv'), dtype='str', delimiter=',')

			if isosbestic_control==False:
				storesList = add_control_channel(filepath, storesList)
				

			if cond==True:
				timestampCorrection_tdt(filepath, timeForLightsTurnOn, storesList)
			else:
				timestampCorrection_csv(filepath, timeForLightsTurnOn, storesList)

			for k in range(storesList.shape[1]):
				decide_naming_convention_and_applyCorrection(filepath, timeForLightsTurnOn, 
															 storesList[0,k], storesList[1,k], storesList)

			# check if isosbestic control is false and also if new control channel is added
			if isosbestic_control==False:
				create_control_channel(filepath, storesList, window=101)




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
		temp = sorted(list(np.concatenate(temp).flatten()), key=str.casefold)
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
def execute_zscore(folderNames, inputParameters):

	timeForLightsTurnOn = inputParameters['timeForLightsTurnOn']
	remove_artifacts = inputParameters['removeArtifacts']
	artifactsRemovalMethod = inputParameters['artifactsRemovalMethod']
	plot_zScore_dff = inputParameters['plot_zScore_dff']
	combine_data = inputParameters['combine_data']
	isosbestic_control = inputParameters['isosbestic_control']

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
			if artifactsRemovalMethod=='concatenate':
				processTimestampsForArtifacts(filepath, timeForLightsTurnOn, storesList)
			else:
				addingNaNtoChunksWithArtifacts(filepath, storesList)
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

	plt.show()
	print("Signal data and event timestamps are extracted.")


def extractTsAndSignal(inputParameters):

	print("Extracting signal data and event timestamps...")

	inputParameters = inputParameters

	#storesList = np.genfromtxt(inputParameters['storesListPath'], dtype='str', delimiter=',')

	folderNames = inputParameters['folderNames']
	timeForLightsTurnOn = inputParameters['timeForLightsTurnOn']
	remove_artifacts = inputParameters['removeArtifacts']
	plot_zScore_dff = inputParameters['plot_zScore_dff']
	combine_data = inputParameters['combine_data']
	isosbestic_control = inputParameters['isosbestic_control']


	print("Remove Artifacts : ", remove_artifacts)
	print("Combine Data : ", combine_data)
	print("Isosbestic Control Channel : ", isosbestic_control)

	if combine_data==False:
		execute_timestamp_correction(folderNames, timeForLightsTurnOn, isosbestic_control)
		execute_zscore(folderNames, inputParameters)
	else:
		execute_timestamp_correction(folderNames, timeForLightsTurnOn, isosbestic_control)
		storesList = check_storeslistfile(folderNames)
		op_folder = combineData(folderNames, timeForLightsTurnOn, storesList)
		execute_zscore(op_folder, inputParameters)
		

	
if __name__ == "__main__":
	extractTsAndSignal(json.loads(sys.argv[1]))



			