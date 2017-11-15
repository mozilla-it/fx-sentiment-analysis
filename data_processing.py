from support_functions import *

target_folder_path = 'Data/2017_11_14/'
date_threshold = '2017-11-01'
col_names = ['Store','Source','Date','Version','Rating','Original Reviews','Translated Reviews','Sentiment','Components','Features','Keywords']
df = data_processing(col_names,target_folder_path,date_threshold)