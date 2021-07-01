import os
import glob
import json
import numpy as np 
import h5py


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

def write_hdf5(data, event, filepath, key):
	op = os.path.join(filepath, event+'.hdf5')
	
	if not os.path.exists(op):
		with h5py.File(op, 'w') as f:
			if type(data) is np.ndarray:
				f.create_dataset(key, data=data, maxshape=(None,), chunks=True)
			else:
				f.create_dataset(key, data=data)
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
				f.create_dataset(key, data=data, maxshape=(None,), chunks=True)


def decide_naming_convention(filepath):
	path_1 = glob.glob(os.path.join(filepath, 'control*'))
	
	path_2 = glob.glob(os.path.join(filepath, 'signal*'))
	
	path = sorted(path_1 + path_2)
	if len(path)%2 != 0:
		raise Exception('There are not equal number of Control and Signal data')
	
	path = np.asarray(path).reshape(2,-1)

	return path


def eliminateData(filepath, timeForLightsTurnOn, event, sampling_rate, naming):
	
	arr = np.array([])
	ts_arr = np.array([])
	for i in range(len(filepath)):
		ts = read_hdf5('timeCorrection_'+naming, filepath[i], 'timestampNew')
		data = read_hdf5(event, filepath[i], 'data').reshape(-1)

		#index = np.where((ts>coords[i,0]) & (ts<coords[i,1]))[0]

		if len(arr)==0:
			arr = np.concatenate((arr, data))
			sub = ts[0]-timeForLightsTurnOn
			new_ts = ts-sub
			ts_arr = np.concatenate((ts_arr, new_ts))
		else:
			temp = data
			temp_ts = ts
			new_ts = temp_ts - (temp_ts[0]-ts_arr[-1])
			arr = np.concatenate((arr, temp))
			ts_arr = np.concatenate((ts_arr, new_ts+(1/sampling_rate)))
	
	return arr, ts_arr


def eliminateTs(filepath, timeForLightsTurnOn, event, sampling_rate, naming):

	ts_arr = np.array([])
	tsNew_arr = np.array([])
	for i in range(len(filepath)):
		tsNew = read_hdf5('timeCorrection_'+naming, filepath[i], 'timestampNew')
		if os.path.exists(os.path.join(filepath[i], event+'_'+naming+'.hdf5')):
			ts = read_hdf5(event+'_'+naming, filepath[i], 'ts').reshape(-1)
		else:
			ts = np.array([])
		
		#print("total time : ", tsNew[-1])
		if len(tsNew_arr)==0:
			sub = tsNew[0]-timeForLightsTurnOn
			tsNew_arr = np.concatenate((tsNew_arr, tsNew-sub))
			ts_arr = np.concatenate((ts_arr, ts-sub))
		else:
			temp_tsNew = tsNew
			temp_ts = ts
			new_ts = temp_ts - (temp_tsNew[0]-tsNew_arr[-1])
			new_tsNew = temp_tsNew - (temp_tsNew[0]-tsNew_arr[-1])
			tsNew_arr = np.concatenate((tsNew_arr, new_tsNew+(1/sampling_rate)))
			ts_arr = np.concatenate((ts_arr, new_ts+(1/sampling_rate)))
		
		#print(event)
		#print(ts_arr)
	return ts_arr

def processTimestampsForCombiningData(filepath, timeForLightsTurnOn, events, sampling_rate):
	
	print("Processing timestamps for combining data...")

	storesList = events[1,:]
	
	for k in range(len(filepath)):

		path = decide_naming_convention(filepath[k][0])
		
		for j in range(path.shape[1]):
			name_1 = ((os.path.basename(path[0,j])).split('.')[0]).split('_')
			name_2 = ((os.path.basename(path[1,j])).split('.')[0]).split('_')
			#dirname = os.path.dirname(path[i])
			if name_1[-1]==name_2[-1]:
				name = name_1[-1]

				for i in range(len(storesList)):
					if 'control_'+name.lower() in storesList[i].lower() or 'signal_'+name.lower() in storesList[i].lower():
						data, timestampNew = eliminateData(filepath[k], timeForLightsTurnOn, storesList[i], sampling_rate, name)
						write_hdf5(data, storesList[i], filepath[k][0], 'data')
					else:
						if 'control' in storesList[i].lower() or 'signal' in storesList[i].lower():
							continue
						else:
							ts = eliminateTs(filepath[k], timeForLightsTurnOn, storesList[i], sampling_rate, name)
							write_hdf5(ts, storesList[i]+'_'+name, filepath[k][0], 'ts')



				write_hdf5(timestampNew, 'timeCorrection_'+name, filepath[k][0], 'timestampNew')

			else:
				raise Exception('Error in naming convention of files or Error in storesList file')

		np.savetxt(os.path.join(filepath[k][0], 'combine_storesList.csv'), events, delimiter=",", fmt='%s')

	print("Timestamps processed and data is combined.")


































	