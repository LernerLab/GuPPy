import os
import sys
import re
import json
import time
import glob
import h5py
import warnings
import logging
from itertools import repeat
import numpy as np
import pandas as pd
from numpy import int32, uint32, uint8, uint16, float64, int64, int32, float32
import multiprocessing as mp
from tqdm import tqdm
from pprint import pprint

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

def writeToFile(value: str):
	with open(os.path.join(os.path.expanduser('~'), 'pbSteps.txt'), 'a') as file:
		file.write(value)

# functino to read tsq file
def readtsq(filepath):
	print("Trying to read tsq file.")
	insertLog("Trying to read tsq file.", logging.DEBUG)
	names = ('size', 'type', 'name', 'chan', 'sort_code', 'timestamp',
	     	'fp_loc', 'strobe', 'format', 'frequency')
	formats = (int32, int32, 'S4', uint16, uint16, float64, int64,
	           float64, int32, float32)
	offsets = 0, 4, 8, 12, 14, 16, 24, 24, 32, 36
	tsq_dtype = np.dtype({'names': names, 'formats': formats,
	                      'offsets': offsets}, align=True)
	path = glob.glob(os.path.join(filepath, '*.tsq'))
	if len(path)>1:
		insertLog('Two tsq files are present at the location.',
	    		  logging.ERROR)
		raise Exception('Two tsq files are present at the location.')
	elif len(path)==0:
		insertLog("\033[1m"+"tsq file not found."+"\033[1m", logging.INFO)
		print("\033[1m"+"tsq file not found."+"\033[1m")
		return 0, 0
	else:
		path = path[0]
		flag = 'tsq'

	# reading tsq file
	tsq = np.fromfile(path, dtype=tsq_dtype)

	# creating dataframe of the data
	df = pd.DataFrame(tsq)

	insertLog("Data from tsq file fetched.", logging.INFO)
	print("Data from tsq file fetched.")
	return df, flag

# function to check if doric file exists
def check_doric(filepath):
	print("Checking if doric file exists.")
	insertLog('Checking if doric file exists', logging.DEBUG)
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
		insertLog('Two doric files are present at the same location', logging.ERROR)
		raise Exception('Two doric files are present at the same location')
	if len(flag_arr)==0:
		insertLog("\033[1m"+"Doric file not found."+"\033[1m", logging.ERROR)
		print("\033[1m"+"Doric file not found."+"\033[1m")
		return 0
	insertLog('Doric file found.', logging.INFO)
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
	insertLog("\033[1m"+"Trying to read data for {} from csv file.".format(event)+"\033[0m", 
	   			logging.DEBUG)
	if not os.path.exists(os.path.join(filepath, event+'.csv')):
		insertLog("\033[1m"+"No csv file found for event {}".format(event)+"\033[0m", 
	    			logging.ERROR)
		raise Exception("\033[1m"+"No csv file found for event {}".format(event)+"\033[0m")

	df = pd.read_csv(os.path.join(filepath, event+'.csv'), index_col=False)
	data = df
	key = list(df.columns)

	if len(key)==3:
		arr1 = np.array(['timestamps', 'data', 'sampling_rate'])
		arr2 = np.char.lower(np.array(key))
		if (np.sort(arr1)==np.sort(arr2)).all()==False:
			insertLog("\033[1m"+"Column names should be timestamps, data and sampling_rate"+"\033[0m",
	     				logging.ERROR)
			raise Exception("\033[1m"+"Column names should be timestamps, data and sampling_rate"+"\033[0m")

	if len(key)==1:
		if key[0].lower()!='timestamps':
			insertLog("\033[1m"+"Column names should be timestamps, data and sampling_rate"+"\033[0m",
	     				logging.ERROR)
			raise Exception("\033[1m"+"Column name should be timestamps"+"\033[0m")

	if len(key)!=3 and len(key)!=1:
		insertLog("\033[1m"+"Number of columns in csv file should be either three or one. Three columns if \
						the file is for control or signal data or one column if the file is for event TTLs."+"\033[0m",
						logging.ERROR)
		raise Exception("\033[1m"+"Number of columns in csv file should be either three or one. Three columns if \
						the file is for control or signal data or one column if the file is for event TTLs."+"\033[0m")
		
	for i in range(len(key)):
		write_hdf5(data[key[i]].dropna(), event, outputPath, key[i].lower())

	insertLog("\033[1m"+"Reading data for {} from csv file is completed.".format(event)+"\033[0m",
	   			logging.INFO)
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

	storesList = np.genfromtxt(os.path.join(outputPath, 'storesList.csv'), dtype='str', delimiter=',').reshape(2,-1)
	
	if diff.shape[0]==0:
		return 0
	
	if S['sampling_rate']==0 and np.all(diff==diff[0])==False:
		print("\033[1m"+"Data in event {} belongs to multiple behavior".format(event)+"\033[0m")
		print("\033[1m"+"Create timestamp files for individual new event and change the stores list file."+"\033[0m")
		insertLog("\033[1m"+"Data in event {} belongs to multiple behavior".format(event)+"\033[0m",
	    			logging.INFO)
		insertLog("\033[1m"+"Create timestamp files for individual new event and change the stores list file."+"\033[0m",
	    			logging.DEBUG)
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
		insertLog("\033[1m"+"Timestamp files for individual new event are created \
	    			and the stores list file is changed."+"\033[0m",
	    			logging.INFO)

			

# function to read tev file
def readtev(data, filepath, event, outputPath):

	print("Reading data for event {} ...".format(event))
	insertLog("Reading data for event {} ...".format(event), logging.DEBUG)
	tevfilepath = glob.glob(os.path.join(filepath, '*.tev'))
	if len(tevfilepath)>1:
		raise Exception('Two tev files are present at the location.')
	else:
		tevfilepath = tevfilepath[0]


	data['name'] = np.asarray(data['name'], dtype=str)

	allnames = np.unique(data['name'])

	index = []
	for i in range(len(allnames)):
		length = len(str(allnames[i]))
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

	S['storename'] = str(event)
	S['sampling_rate'] = data['frequency'][first_row]
	S['timestamps'] = np.asarray(data['timestamp'][allIndexesWhereEventIsPresent[0]])
	S['channels'] = np.asarray(data['chan'][allIndexesWhereEventIsPresent[0]])


	fp_loc = np.asarray(data['fp_loc'][allIndexesWhereEventIsPresent[0]])
	data_size = np.asarray(data['size'])

	if formatNew != 5:
		nsample = (data_size[first_row,]-10)*int(table[formatNew, 2])
		S['data'] = np.zeros((len(fp_loc), nsample))
		for i in tqdm(range(0, len(fp_loc))):
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

	S_print = S.copy()
	S_print.pop('data')
	pprint(S_print)
	
	save_dict_to_hdf5(S, event, outputPath)
	
	check_data(S, filepath, event, outputPath)

	print("Data for event {} fetched and stored.".format(event))
	insertLog("Data for event {} fetched and stored.".format(event), logging.INFO)


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

def access_data_doricV1(doric_file, storesList, outputPath):
	keys = list(doric_file['Traces']['Console'].keys())
	for i in range(storesList.shape[1]):
		if 'control' in storesList[1,i] or 'signal' in storesList[1,i]:
			timestamps = np.array(doric_file['Traces']['Console']['Time(s)']['Console_time(s)'])
			sampling_rate = np.array([1/(timestamps[-1]-timestamps[-2])])
			data = np.array(doric_file['Traces']['Console'][storesList[0,i]][storesList[0,i]])
			write_hdf5(sampling_rate, storesList[0,i], outputPath, 'sampling_rate')
			write_hdf5(timestamps, storesList[0,i], outputPath, 'timestamps')
			write_hdf5(data, storesList[0,i], outputPath, 'data')
		else:
			timestamps = np.array(doric_file['Traces']['Console']['Time(s)']['Console_time(s)'])
			ttl = np.array(doric_file['Traces']['Console'][storesList[0,i]][storesList[0,i]])
			indices = np.where(ttl<=0)[0]
			diff_indices = np.where(np.diff(indices)>1)[0]
			write_hdf5(timestamps[indices[diff_indices]+1], storesList[0,i], outputPath, 'timestamps')

def separate_last_element(arr):
    l = arr[-1]
    return arr[:-1], l

def find_string(regex, arr):
	for i in range(len(arr)):
		if regex.match(arr[i]):
			return i

def access_data_doricV6(doric_file, storesList, outputPath):
	data = [doric_file["DataAcquisition"]]
	res = []
	while len(data) != 0:
		members = len(data)
		while members != 0:
			members -= 1
			data, last_element = separate_last_element(data)
			if isinstance(last_element, h5py.Dataset) and not last_element.name.endswith("/Time"):
				res.append(last_element.name)
			elif isinstance(last_element, h5py.Group):
				data.extend(reversed([last_element[k] for k in last_element.keys()]))
	
	decide_path = []
	for element in res:
		sep_values = element.split('/')
		if sep_values[-1]=='Values':
			if sep_values[-2] in storesList[0,:]:
				decide_path.append(element)
		else:
			if sep_values[-1] in storesList[0,:]:
				decide_path.append(element)
	
	for i in range(storesList.shape[1]):
		if 'control' in storesList[1,i] or 'signal' in storesList[1,i]:
			regex = re.compile('(.*?)'+str(storesList[0,i])+'(.*?)')
			idx = [i for i in range(len(decide_path)) if regex.match(decide_path[i])]
			if len(idx)>1:
				insertLog()
				raise Exception('More than one string matched (which should not be the case)',
		    					logging.ERROR)
			idx = idx[0]
			data = np.array(doric_file[decide_path[idx]])
			timestamps = np.array(doric_file[decide_path[idx].rsplit('/',1)[0]+'/Time'])
			sampling_rate = np.array([1/(timestamps[-1]-timestamps[-2])])
			write_hdf5(sampling_rate, storesList[0,i], outputPath, 'sampling_rate')
			write_hdf5(timestamps, storesList[0,i], outputPath, 'timestamps')
			write_hdf5(data, storesList[0,i], outputPath, 'data')
		else:
			regex = re.compile('(.*?)'+storesList[0,i]+'$')
			idx = [i for i in range(len(decide_path)) if regex.match(decide_path[i])]
			if len(idx)>1:
				insertLog('More than one string matched (which should not be the case)',
	      					logging.ERROR)
				raise Exception('More than one string matched (which should not be the case)')
			idx = idx[0]
			ttl = np.array(doric_file[decide_path[idx]])
			timestamps = np.array(doric_file[decide_path[idx].rsplit('/',1)[0]+'/Time'])
			indices = np.where(ttl<=0)[0]
			diff_indices = np.where(np.diff(indices)>1)[0]
			write_hdf5(timestamps[indices[diff_indices]+1], storesList[0,i], outputPath, 'timestamps')

def execute_import_doric(filepath, storesList, flag, outputPath):
	
	if flag=='doric_csv':
		path = glob.glob(os.path.join(filepath, '*.csv'))
		if len(path)>1:
			insertLog('An error occurred : More than one Doric csv file present at the location',
	     				logging.ERROR)
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
			insertLog('An error occurred : More than one Doric file present at the location',
	     				logging.ERROR)
			raise Exception('More than one Doric file present at the location')
		else:
			with h5py.File(path[0], 'r') as f:
				if 'Traces' in list(f.keys()):
					keys = access_data_doricV1(f, storesList, outputPath)
				elif list(f.keys())==['Configurations', 'DataAcquisition']:
					keys = access_data_doricV6(f, storesList, outputPath)
				

# function to read data from 'tsq' and 'tev' files
def readRawData(inputParametersPath):

	
	print('### Reading raw data... ###')
	insertLog('### Reading raw data... ###', logging.DEBUG)
	# get input parameters
	with open(inputParametersPath) as f:	
		inputParameters = json.load(f)

	folderNames = inputParameters['folderNames']
	numProcesses = inputParameters['numberOfCores']
	storesListPath = []
	if numProcesses==0:
		numProcesses = mp.cpu_count()
	elif numProcesses>mp.cpu_count():
		print('Warning : # of cores parameter set is greater than the cores available \
			   available in your machine')
		insertLog('Warning : # of cores parameter set is greater than the cores available \
			   available in your machine', logging.WARNING)
		numProcesses = mp.cpu_count()-1
	for i in range(len(folderNames)):
		filepath = folderNames[i]
		storesListPath.append(glob.glob(os.path.join(filepath, '*_output_*')))
	storesListPath = np.concatenate(storesListPath)
	writeToFile(str((storesListPath.shape[0]+1)*10)+'\n'+str(10)+'\n')
	step = 0
	for i in range(len(folderNames)):
		filepath = folderNames[i]
		print(filepath)
		insertLog(f"### Reading raw data for folder {folderNames[i]}", logging.DEBUG)
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
				storesList = np.genfromtxt(os.path.join(op, '.cache_storesList.csv'), dtype='str', delimiter=',').reshape(2,-1)
			else:
				storesList = np.genfromtxt(os.path.join(op, 'storesList.csv'), dtype='str', delimiter=',').reshape(2,-1)

			if isinstance(data, pd.DataFrame) and flag=='tsq':
				execute_readtev(data, filepath, np.unique(storesList[0,:]), op, numProcesses)
			elif flag=='doric_csv':
				execute_import_doric(filepath, storesList, flag, op)
			elif flag=='doric_doric':
				execute_import_doric(filepath, storesList, flag, op)
			else:
				execute_import_csv(filepath, np.unique(storesList[0,:]), op, numProcesses)

			writeToFile(str(10+((step+1)*10))+'\n')
			step += 1
		insertLog(f"### Raw data for folder {folderNames[i]} fetched", logging.INFO)
	print("### Raw data fetched and saved.")
	insertLog('Raw data fetched and saved.', logging.INFO)
	insertLog("#" * 400, logging.INFO)

# from pynwb import NWBHDF5IO
# def read_nwb(filepath, event, outputPath, indices):
# 	"""
# 	Read photometry data from an NWB file and save the output to a hdf5 file.
# 	"""
# 	print(f"Reading NWB file {filepath} for event {event} to save to {outputPath} with indices {indices}")

# 	with NWBHDF5IO(filepath, 'r') as io:
# 		nwbfile = io.read()
# 		fiber_photometry_response_series = nwbfile.acquisition[event].data[:, indices]
# 		sampling_rate = fiber_photometry_response_series.rate

# 	S = dict()
# 	S['storename'] = str(event)
# 	S['sampling_rate'] = sampling_rate
# 	S['timestamps'] = np.arange(0, fiber_photometry_response_series.shape[0]) / sampling_rate
# 	S['data'] = fiber_photometry_response_series
    # save_dict_to_hdf5(S, event, outputPath)
    # check_data(S, filepath, event, outputPath)
    # print("Data for event {} fetched and stored.".format(event))
    # insertLog("Data for event {} fetched and stored.".format(event), logging.INFO)

# if __name__ == "__main__":
# 	print('run')
# 	try:
# 		readRawData(json.loads(sys.argv[1]))
# 		insertLog('#'*400, logging.INFO)
# 	except Exception as e:
# 		with open(os.path.join(os.path.expanduser('~'), 'pbSteps.txt'), 'a') as file:
# 			file.write(str(-1)+"\n")
# 		insertLog(f"An error occurred: {e}", logging.ERROR)
# 		raise e

