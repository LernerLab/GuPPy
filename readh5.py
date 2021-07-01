import pandas as pd


# filepath where your file is located 
filepath = '/Users/VENUS/Downloads/JIllian/Abby/average/peak_AUC_UnrewardedPort_DLS_z_score_DLS.h5'

# read file and make a dataframe
df = pd.read_hdf(filepath, key='df', mode='r')

# print dataframe
print(df)