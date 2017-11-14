import numpy as np
import pandas as pd

def preprocess_cate(cate_file_path):
    """
    Function to pre-process the categorization data
    """
    cate_file = pd.read_csv(cate_file_path)
    cate_file = cate_file.iloc[:,0:2]
    cate_file = cate_file.dropna(axis=0, how='all')
    component_prior = 'Unknown'
    for i in range(len(cate_file)):
        if pd.isnull(cate_file.iloc[i,0]):
            cate_file.iloc[i,0] = component_prior
        else:
            component_prior = cate_file.iloc[i,0]
    cate_file.to_csv(cate_file_path, index=False)
    return cate_file

cate_file_path = 'Data/Categorization.csv'
pre_processing = False

if pre_processing:
    cate_file = preprocess_cate(cate_file_path)
else:
    cate_file = pd.read_csv(cate_file_path)