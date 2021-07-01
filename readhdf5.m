% filepath to the control and signal data which needs to be read
filepath_data = '/Users/VENUS/Downloads/controlA.hdf5';

% To read control and signal data, use the following syntax
data = h5read(filepath_data, '/data');

% filepath to control and signal data timestamps which needs to be read
filepath_timestamps = '/Users/VENUS/Downloads/timeCorrection_A.hdf5';

% To read control and signal data timestamps, use the following syntax 
timestamps = h5read(filepath_timestamps, '/timestampNew');

% filepath to event timestamps which needs to be read
filepath_event = '/Users/VENUS/Downloads/rwdNP_A.hdf5';

% To read event timestamps files, use the following syntax
ts = h5read(filepath_event, '/ts');
