import os
import json
import glob
import h5py
import numpy as np

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

def decide_block_ts(filepath, storesList, 
					block_change_interval=20, number_of_blocks = 3,
					combination=['LueR','LueU','RueR', 'RueU']):

	storenames = storesList[0,:]
	storesList = storesList[1,:]

	check_combination = np.intersect1d(storenames, combination)
	if check_combination.shape[0]!=len(combination):
		raise Exception('Storenames do not include any or all of these values {}'.format(combination))
	elif (check_combination==combination).all():
		pass
	else:
		raise Exception('Storenames do not include any or all of these values {}'.format(combination))

	arr = []
	for i in range(storesList.shape[0]):
	    if 'control' in storesList[i].lower() or 'signal' in storesList[i].lower():
	        arr.append(storesList[i].split('_')[-1])
	arr = list(set(arr))

	combine_ts = list()
	for i in range(storenames.shape[0]):
		if storenames[i] in combination:
			combine_ts.append(read_hdf5(storesList[i]+'_'+arr[0], filepath, 'ts'))

	combine_ts = sorted(np.concatenate(combine_ts))
	block_change_ts = []
	ix = 1
	while ix<number_of_blocks:
		index = (ix*block_change_interval)-1
		ix+=1
		if index<len(combine_ts):
			block_change_ts.append(combine_ts[index])
		else:
			continue
	
	block_change_ts = [0.] + block_change_ts + [99999.]
	new_storeslist, new_storenames = list(), list()
	for i in range(storesList.shape[0]):
		if 'control' in storesList[i].lower() or 'signal' in storesList[i].lower():
			continue
		else:
			for naming in arr:
				ts = read_hdf5(storesList[i]+'_'+naming, filepath, 'ts')
				for j in range(1, len(block_change_ts)):
					block_ts = ts[np.where((ts>block_change_ts[j-1]) & (ts<=block_change_ts[j]))[0]]
					if block_ts.shape[0]>0:
						write_hdf5(block_ts, '{}_block_{}_{}'.format(storesList[i], j, naming), filepath, 'ts')
						if '{}_block_{}'.format(storenames[i], j) not in new_storenames:
							new_storenames.append('{}_block_{}'.format(storenames[i], j))
						else:
							continue
						if '{}_block_{}'.format(storesList[i], j) not in new_storeslist:
							new_storeslist.append('{}_block_{}'.format(storesList[i], j))
						else:
							continue

					else:
						continue

	
	for i in range(storesList.shape[0]):
		if 'control' in storesList[i].lower() or 'signal' in storesList[i].lower():
			new_storenames.append(storenames[i])
			new_storeslist.append(storesList[i])
		else:
			continue	
	np.savetxt(os.path.join(filepath, 'storesList.csv'), 
			   np.array([new_storenames, new_storeslist]), 
			   delimiter=",", fmt='%s')
	


def execute_splitTsBlocks(folderNames):

	print('Splitting timestamps based on blocks')
	for i in range(len(folderNames)):
		filepath = folderNames[i]
		storesListPath = glob.glob(os.path.join(filepath, '*_output_*'))
		for j in range(len(storesListPath)):
			filepath = storesListPath[j]
			storesList = np.genfromtxt(os.path.join(filepath, 'storesList.csv'), dtype='str', delimiter=',')

			decide_block_ts(filepath, storesList)

	print('Splitting of timestamps is completed.')

def splitTsBlocks(inputParametersPath):

	with open(inputParametersPath) as f:	
		inputParameters = json.load(f)

	folderNames = inputParameters['folderNames']
	execute_splitTsBlocks(folderNames)
