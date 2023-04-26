import os
import numpy as np
import h5py


# path to z-score file
fp = "/Users/VENUS/Downloads/FP_Data/habitEarly/Photo_63_207-181030-103332/Photo_63_207-181030-103332_output_1/z_score_DMS.hdf5"

with h5py.File(fp, 'r') as f:

	# print keys in hdf5 files
	keys = list(f.keys())
	print(list(f.keys()))

	# create a data dictionary
	data = dict()

	# loop through each key and save the data corresponding to a key in a dictionary
	for key in keys:
		data[key] = np.array(f[key])

