import os
import sys
import json
import time
import glob
import h5py
import warnings
from itertools import repeat
import numpy as np
import pandas as pd
from numpy import int32, uint32, uint8, uint16, float64, int64, int32, float32
import multiprocessing as mp


# functino to read tsq file
def readtsq(filepath):
	print("### Trying to read tsq file....")
	names = ('size', 'type', 'name', 'chan', 'sort_code', 'timestamp',
	     	'fp_loc', 'strobe', 'format', 'frequency')
	formats = (int32, int32, 'S4', uint16, uint16, float64, int64,
	           float64, int32, float32)
	offsets = 0, 4, 8, 12, 14, 16, 24, 24, 32, 36
	tsq_dtype = np.dtype({'names': names, 'formats': formats,
	                      'offsets': offsets}, align=True)
	path = glob.glob(os.path.join(filepath, '*.tsq'))
	if len(path)>1:
		raise Exception('Two tsq files are present at the location.')
	elif len(path)==0:
		print("\033[1m"+"tsq file not found."+"\033[1m")
		return 0, 0
	else:
		path = path[0]
		flag = 'tsq'

	# reading tsq file
	tsq = np.fromfile(path, dtype=tsq_dtype)

	# creating dataframe of the data
	df = pd.DataFrame(tsq)

	print("Data from tsq file fetched....")
	return df, flag

# function to check if doric file exists
def check_doric(filepath):
	print("### Checking if doric file exists...")
	path = glob.glob(os.path.join(filepath, '*.csv')) + \
		   glob.glob(os.path.join(filepath, '*.doric'))
	
	flag_arr = []
	for i in range(len(path)):
		ext = os.path.basename(path[i]).split('.')[-1]
		if ext=='csv':
			with warnings.catch_warnings():
				warnings.simplefilter("error")
				try:
					df = pd.read_csv(path[i], index_col=False, dtype=float)
				except:
					df = pd.read_csv(path[i], header=1, index_col=False, nrows=10)
					flag = 'doric_csv'
					flag_arr.append(flag)
		elif ext=='doric':
			flag = 'doric_doric'
			flag_arr.append(flag)
		else:
			pass

	if len(flag_arr)>1:
		raise Exception('Two doric files are present at the same location')
	if len(flag_arr)==0:
		print("\033[1m"+"Doric file not found."+"\033[1m")
		return 0
	print('Doric file found.')
	return flag_arr[0]
		

# check if a particular element is there in an array or not
def ismember(arr, element):
    res = [1 if i==element else 0 for i in arr]
    return np.asarray(res)


# function to write data to a hdf5 file
def write_hdf5(data, event, filepath, key):

	# replacing \\ or / in storenames with _ (to avoid errors while saving data)
	event = event.replace("\\","_")
	event = event.replace("/","_")

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


# function to read event timestamps csv file.
def import_csv(filepath, event, outputPath):
	print("\033[1m"+"Trying to read data for {} from csv file.".format(event)+"\033[0m")
	if not os.path.exists(os.path.join(filepath, event+'.csv')):
		raise Exception("\033[1m"+"No csv file found for event {}".format(event)+"\033[0m")

	df = pd.read_csv(os.path.join(filepath, event+'.csv'), index_col=False)
	data = df
	key = list(df.columns)

	if len(key)==3:
		arr1 = np.array(['timestamps', 'data', 'sampling_rate'])
		arr2 = np.char.lower(np.array(key))
		if (np.sort(arr1)==np.sort(arr2)).all()==False:
			raise Exception("\033[1m"+"Column names should be timestamps, data and sampling_rate"+"\033[0m")

	if len(key)==1:
		if key[0].lower()!='timestamps':
			raise Exception("\033[1m"+"Column name should be timestamps"+"\033[0m")

	if len(key)!=3 and len(key)!=1:
		raise Exception("\033[1m"+"Number of columns in csv file should be either three or one. Three columns if \
								   the file is for control or signal data or one column if the file is for event TTLs."+"\033[0m")
		
	for i in range(len(key)):
		write_hdf5(data[key[i]].dropna(), event, outputPath, key[i].lower())

	print("\033[1m"+"Reading data for {} from csv file is completed.".format(event)+"\033[0m")

	return data, key

# function to save data read from tev file to hdf5 file
def save_dict_to_hdf5(S, event, outputPath):
	write_hdf5(S['storename'], event, outputPath, 'storename')
	write_hdf5(S['sampling_rate'], event, outputPath, 'sampling_rate')
	write_hdf5(S['timestamps'], event, outputPath, 'timestamps')

	write_hdf5(S['data'], event, outputPath, 'data')
	write_hdf5(S['npoints'], event, outputPath, 'npoints')
	write_hdf5(S['channels'], event, outputPath, 'channels')



# function to check event data (checking whether event timestamps belongs to same event or multiple events)
def check_data(S, filepath, event, outputPath):
	#print("Checking event storename data for creating multiple event names from single event storename...")
	new_event = event.replace("\\","")
	new_event = event.replace("/","")
	diff = np.diff(S['data'])
	arr = np.full(diff.shape[0],1)

	storesList = np.genfromtxt(os.path.join(outputPath, 'storesList.csv'), dtype='str', delimiter=',')
	
	if diff.shape[0]==0:
		return 0
	
	if S['sampling_rate']==0 and np.all(diff==diff[0])==False:
		print("\033[1m"+"Data in event {} belongs to multiple behavior".format(event)+"\033[0m")
		print("\033[1m"+"Create timestamp files for individual new event and change the stores list file."+"\033[0m")
		i_d = np.unique(S['data'])
		for i in range(i_d.shape[0]):
			new_S = dict()
			idx = np.where(S['data']==i_d[i])[0]
			new_S['timestamps'] = S['timestamps'][idx]
			new_S['storename'] = new_event+str(int(i_d[i]))
			new_S['sampling_rate'] = S['sampling_rate']
			new_S['data'] = S['data']
			new_S['npoints'] = S['npoints']
			new_S['channels'] = S['channels']
			storesList = np.concatenate((storesList, [[new_event+str(int(i_d[i]))], [new_event+'_'+str(int(i_d[i]))]]), axis=1)
			save_dict_to_hdf5(new_S, new_event+str(int(i_d[i])), outputPath)

		idx = np.where(storesList[0]==event)[0]
		storesList = np.delete(storesList,idx,axis=1)
		if not os.path.exists(os.path.join(outputPath, '.cache_storesList.csv')):
			os.rename(os.path.join(outputPath, 'storesList.csv'), os.path.join(outputPath, '.cache_storesList.csv'))
		if idx.shape[0]==0:
			pass 
		else:
			np.savetxt(os.path.join(outputPath, 'storesList.csv'), storesList, delimiter=",", fmt='%s')

			

# function to read tev file
def readtev(data, filepath, event, outputPath):

	print("Reading data for event {} ...".format(event))

	tevfilepath = glob.glob(os.path.join(filepath, '*.tev'))
	if len(tevfilepath)>1:
		raise Exception('Two tev files are present at the location.')
	else:
		tevfilepath = tevfilepath[0]


	data['name'] = np.asarray(data['name'], dtype=np.str)

	allnames = np.unique(data['name'])

	index = []
	for i in range(len(allnames)):
		length = len(np.str(allnames[i]))
		if length<4:
			index.append(i)


	allnames = np.delete(allnames, index, 0)


	eventNew = np.array(list(event))

	#print(allnames)
	#print(eventNew)
	row = ismember(data['name'], event)


	if sum(row)==0:
		print("\033[1m"+"Requested store name "+event+" not found (case-sensitive)."+"\033[0m")
		print("\033[1m"+"File contains the following TDT store names:"+"\033[0m")
		print("\033[1m"+str(allnames)+"\033[0m")
		print("\033[1m"+"TDT store name "+str(event)+" not found."+"\033[0m")
		import_csv(filepath, event, outputPath)

		return 0
		
	allIndexesWhereEventIsPresent = np.where(row==1)
	first_row = allIndexesWhereEventIsPresent[0][0]

	formatNew = data['format'][first_row]+1

	table = np.array([[0,0,0,0],
	                    [0,'float',1, np.float32],
	                     [0,'long', 1, np.int32],
	                 [0,'short',2, np.int16], 
	                 [0,'byte', 4, np.int8]])

	S = dict()

	S['storename'] = np.str(event)
	S['sampling_rate'] = data['frequency'][first_row]
	S['timestamps'] = np.asarray(data['timestamp'][allIndexesWhereEventIsPresent[0]])
	S['channels'] = np.asarray(data['chan'][allIndexesWhereEventIsPresent[0]])


	fp_loc = np.asarray(data['fp_loc'][allIndexesWhereEventIsPresent[0]])
	data_size = np.asarray(data['size'])

	if formatNew != 5:
		nsample = (data_size[first_row,]-10)*int(table[formatNew, 2])
		S['data'] = np.zeros((len(fp_loc), nsample))
		for i in range(0, len(fp_loc)):
			with open(tevfilepath, 'rb') as fp:
				fp.seek(fp_loc[i], os.SEEK_SET)
				S['data'][i,:] = np.fromfile(fp, dtype=table[formatNew, 3], count=nsample).reshape(1, nsample, order='F')
				#S['data'] = S['data'].swapaxes()
		S['npoints'] = nsample
	else:
		S['data'] = np.asarray(data['strobe'][allIndexesWhereEventIsPresent[0]])
		S['npoints'] = 1
		S['channels'] = np.tile(1, (S['data'].shape[0],))


	S['data'] = (S['data'].T).reshape(-1, order='F')
	
	save_dict_to_hdf5(S, event, outputPath)
	
	check_data(S, filepath, event, outputPath)

	print("Data for event {} fetched and stored.".format(event))


# function to execute readtev function using multiprocessing to make it faster
def execute_readtev(data, filepath, event, outputPath, numProcesses=mp.cpu_count()):

	start = time.time()
	with mp.Pool(numProcesses) as p:
		p.starmap(readtev, zip(repeat(data), repeat(filepath), event, repeat(outputPath)))
	#p = mp.Pool(mp.cpu_count())
	#p.starmap(readtev, zip(repeat(data), repeat(filepath), event, repeat(outputPath)))
	#p.close()
	#p.join()
	print("Time taken = {0:.5f}".format(time.time() - start))


def execute_import_csv(filepath, event, outputPath, numProcesses=mp.cpu_count()):
	#print("Reading data for event {} ...".format(event))

	start = time.time()
	with mp.Pool(numProcesses) as p:
		p.starmap(import_csv, zip(repeat(filepath), event, repeat(outputPath)))
	print("Time taken = {0:.5f}".format(time.time() - start))

def execute_import_doric(filepath, storesList, flag, outputPath):
	
	if flag=='doric_csv':
		path = glob.glob(os.path.join(filepath, '*.csv'))
		if len(path)>1:
			raise Exception('More than one Doric csv file present at the location')
		else:
			df = pd.read_csv(path[0], header=1, index_col=False)
			df = df.dropna(axis=1, how='all')
			df = df.dropna(axis=0, how='any')
			df['Time(s)'] = df['Time(s)'] - df['Time(s)'].to_numpy()[0]	
			for i in range(storesList.shape[1]):
				if 'control' in storesList[1,i] or 'signal' in storesList[1,i]:
					timestamps = np.array(df['Time(s)'])
					sampling_rate = np.array([1/(timestamps[-1]-timestamps[-2])])
					write_hdf5(sampling_rate, storesList[0,i], outputPath, 'sampling_rate')
					write_hdf5(df['Time(s)'].to_numpy(), storesList[0,i], outputPath, 'timestamps')
					write_hdf5(df[storesList[0,i]].to_numpy(), storesList[0,i], outputPath, 'data')
				else:
					ttl = df[storesList[0,i]]
					indices = np.where(ttl<=0)[0]
					diff_indices = np.where(np.diff(indices)>1)[0]
					write_hdf5(df['Time(s)'][indices[diff_indices]+1].to_numpy(), storesList[0,i], outputPath, 'timestamps')
	else:
		path = glob.glob(os.path.join(filepath, '*.doric'))
		if len(path)>1:
			raise Exception('More than one Doric file present at the location')
		else:
			with h5py.File(path[0], 'r') as f:
				keys = list(f['Traces']['Console'].keys())
				for i in range(storesList.shape[1]):
					if 'control' in storesList[1,i] or 'signal' in storesList[1,i]:
						timestamps = np.array(f['Traces']['Console']['Time(s)']['Console_time(s)'])
						sampling_rate = np.array([1/(timestamps[-1]-timestamps[-2])])
						data = np.array(f['Traces']['Console'][storesList[0,i]][storesList[0,i]])
						write_hdf5(sampling_rate, storesList[0,i], outputPath, 'sampling_rate')
						write_hdf5(timestamps, storesList[0,i], outputPath, 'timestamps')
						write_hdf5(data, storesList[0,i], outputPath, 'data')
					else:
						timestamps = np.array(f['Traces']['Console']['Time(s)']['Console_time(s)'])
						ttl = np.array(f['Traces']['Console'][storesList[0,i]][storesList[0,i]])
						indices = np.where(ttl<=0)[0]
						diff_indices = np.where(np.diff(indices)>1)[0]
						write_hdf5(timestamps[indices[diff_indices]+1], storesList[0,i], outputPath, 'timestamps')


# function to read data from 'tsq' and 'tev' files
def readRawData(inputParameters):


	print('### Reading raw data... ###')

	# get input parameters
	inputParameters = inputParameters

	#storesListPath = glob.glob(os.path.join('/Users/VENUS/Downloads/Ashley/', '*_output_*'))

	folderNames = inputParameters['folderNames']
	numProcesses = inputParameters['numberOfCores']
	if numProcesses==0:
		numProcesses = mp.cpu_count()
	elif numProcesses>mp.cpu_count():
		print('Warning : # of cores parameter set is greater than the cores available \
			   available in your machine')
		numProcesses = mp.cpu_count()-1

	for i in folderNames:
		filepath = i
		print(filepath)
		storesListPath = glob.glob(os.path.join(filepath, '*_output_*'))
		# reading tsq file
		data, flag = readtsq(filepath)
		# checking if doric file exists
		if flag=='tsq':
			pass
		else:
			flag = check_doric(filepath)

		# read data corresponding to each storename selected by user while saving the storeslist file
		for j in range(len(storesListPath)):
			op = storesListPath[j]
			if os.path.exists(os.path.join(op, '.cache_storesList.csv')):
				storesList = np.genfromtxt(os.path.join(op, '.cache_storesList.csv'), dtype='str', delimiter=',')
			else:
				storesList = np.genfromtxt(os.path.join(op, 'storesList.csv'), dtype='str', delimiter=',')

			if isinstance(data, pd.DataFrame) and flag=='tsq':
				execute_readtev(data, filepath, np.unique(storesList[0,:]), op, numProcesses)
			elif flag=='doric_csv':
				execute_import_doric(filepath, storesList, flag, op)
			elif flag=='doric_doric':
				execute_import_doric(filepath, storesList, flag, op)
			else:
				execute_import_csv(filepath, np.unique(storesList[0,:]), op, numProcesses)

	print("Raw data fetched and saved.")

if __name__ == "__main__":
	print('run')
	readRawData(json.loads(sys.argv[1]))

