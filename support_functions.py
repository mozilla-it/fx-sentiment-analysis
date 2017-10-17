import numpy as np
import pandas as pd
import os, os.path
import glob
import re

def data_integration(col_names,target_folder_path = '',save_csv = True):
    Appbot, SurveyGizmo = read_data(target_folder_path)
    Appbot_Processed = process_appbot_df(Appbot,col_names)
    SurveyGizmo_Processed = process_surveygizmo_df(SurveyGizmo,col_names)
    df = pd.concat([Appbot_Processed,SurveyGizmo_Processed]) # Merged the dataframes
    
    # Save into an output file in the target folder
    if save_csv:
        output_path = target_folder_path + 'output_py.csv'
        df.to_csv(output_path)
    return df

def read_data(target_folder_path):
    """
    Function to read through all the datasets in the target folder
    todo: add support to the senario where there are multiple Appbot/SurveyGizmo files in the folder
    """
    # Read in all the dataset
    file_paths = glob.glob(target_folder_path + '*')
    for file_path in file_paths:
        # We need to ignore the previously generated output file, which contains 'py' in the end of filename
        if file_path.split('.')[-2][-2:] == 'py': # All of the code-generated file contains 'py' in the end of filename
            os.remove(file_path) # Remove it - we will create a new one
        else:
            file_format = file_path.split('.')[-1]
            if file_format == 'xlsx':
                xl = pd.ExcelFile(file_path)
                SurveyGizmo_df = xl.parse(xl.sheet_names[0]).fillna('')
            else:
                Appbot_df = pd.read_csv(file_path).fillna('')
    return Appbot_df, SurveyGizmo_df

def create_empty_df(n,col_names):
    df = pd.DataFrame('', index=range(n), columns=col_names)
    return df

def extract_version_SG(SG_Col):
    """
    Function to extract the version information from the Corresponding Column in SurveyGizmo
    todo: add support to the format of 9.34 - now we can only extract 9.3
    todo: add support to the format of 9.0.1 - now we can only extracvt 9.0
    """
    version_list = []
    for i in range(len(SG_Col)):
        string = SG_Col[i] # Extract the string in the current row
        locator = string.find("FxiOS/") # Locate the target term in each string
        if locator > 0: # Find the keyword
            version_code = string.split("FxiOS/",1)[1].split(' ')[0]  # Example: 10.0b6373
            version = re.findall("\d+\.\d+", version_code)[0] # Extract the float number in the string
        else:
            version = ''
        version_list.append(version)
    return version_list

def process_appbot_df(Appbot,col_names):
    """
    Function to Process the Appbot Dataframe
    """
    Appbot_Processed = create_empty_df(len(Appbot),col_names) # Initialize a new dataframe
    
    Appbot_Processed['Store'] =  Appbot['Store']
    Appbot_Processed['Source'] =  'Appbot'
    Appbot_Processed['Date'] =  Appbot['Date']
    Appbot_Processed['Version'] =  Appbot['Version']
    Appbot_Processed['Rating'] =  Appbot['Rating']
    Appbot_Processed['Emotion'] =  Appbot['Emotion']
    Appbot_Processed['Original Reviews'] = Appbot[['Subject','Body']].apply(lambda x : '{}. {}'.format(x[0],x[1]), axis=1)
    Appbot_Processed['Translated Reviews'] = Appbot[['Translated Subject','Translated Body']].apply(lambda x : '{}. {}'.format(x[0],x[1]), axis=1)
    
    return Appbot_Processed

def process_surveygizmo_df(SurveyGizmo,col_names):
    """
    Function to Process the SurveyGizmo Dataframe
    """
    SurveyGizmo_Processed = create_empty_df(len(SurveyGizmo),col_names) # Initialize a new dataframe
    
    SurveyGizmo_Processed['Store'] = 'iOS'
    SurveyGizmo_Processed['Source'] = 'Browser'
    SurveyGizmo_Processed['Date'] = pd.to_datetime(SurveyGizmo[SurveyGizmo.columns[0]]).dt.date
    SurveyGizmo_Processed['Version'] = extract_version_SG(SurveyGizmo[SurveyGizmo.columns[3]])
    SurveyGizmo_Processed['Emotion'] = SurveyGizmo[SurveyGizmo.columns[5]]
    SurveyGizmo_Processed['Original Reviews'] = SurveyGizmo[[SurveyGizmo.columns[6],SurveyGizmo.columns[7]]].apply(lambda x : '{}{}'.format(x[0],x[1]), axis=1)
    SurveyGizmo_Processed['Translated Reviews'] = ''
    
    return SurveyGizmo_Processed

