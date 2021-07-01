% filepath to the file which needs to be read
filepath = '/Users/VENUS/Downloads/JIllian/habitEarly/average/peak_AUC_rwdNP_DMS_z_score_DMS.h5';
% access h5 file column names
info_data = h5read(filepath, '/df/axis0');
% access each column data in h5 file
data = h5read(filepath, '/df/block0_values');

% access h5 file row names
info_filename = h5read(filepath, '/df/axis1');

% construct a struct with three keys : filename, freq, amplitude
S.filename = info_filename;

% for reading freq and amplitude from z-score
S.freq = data(1,:)';
S.amplitude = data(2,:)';

% construct table from the above struct
T = struct2table(S);

% To access each column in table use "T.{column_name}"