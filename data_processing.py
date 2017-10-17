from support_functions import *

target_folder_path = 'Data/2017_10_16/'
col_names = ['Store','Source','Date','Version','Emotion','Rating','Original Reviews','Translated Reviews']
df = data_integration(col_names,target_folder_path)