import numpy as np
import pandas as pd
import os, os.path
import glob
import re
from google.cloud import translate
from google.oauth2 import service_account
from langdetect import detect
from datetime import datetime

def data_processing(col_names, target_folder_path,date_threshold = '', save_csv = True):
    df = data_integration(col_names,target_folder_path)
    
    if len(date_threshold) > 0:
        df = filter_by_date(df, date_threshold) # Remove rows whose date is before the given date thershold

    df = translate_reviews(df) # Translate non-English reviews

    # Save into an output file in the target folder
    if save_csv:
        output_path = target_folder_path + 'output_py.xlsx'
        # df.to_csv(output_path,encoding='utf-8')
        df.to_excel(output_path,sheet_name='Sheet1',index= False)
    return df
    
def data_integration(col_names,target_folder_path):
    """
    Function to read, process, and integrate the data
    """
    Appbot, SurveyGizmo = read_data(target_folder_path)
    Appbot_Processed = process_appbot_df(Appbot,col_names)
    SurveyGizmo_Processed = process_surveygizmo_df(SurveyGizmo,col_names)
    df = pd.concat([Appbot_Processed,SurveyGizmo_Processed]) # Merged the dataframes
    
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
            version = re.findall("^\d+\.\d+\.\d+|^\d+\.\d+", version_code)[0] # Extract the float number in the string
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
    Appbot_Processed['Date'] =  pd.to_datetime(Appbot['Date']).dt.date
    Appbot_Processed['Version'] =  Appbot['Version']
    Appbot_Processed['Rating'] =  Appbot['Rating']
    Appbot_Processed['Emotion'] =  Appbot['Emotion']
    Appbot_Processed['Original Reviews'] = Appbot[['Subject','Body']].apply(lambda x : '{}. {}'.format(x[0],x[1]), axis=1)
    Appbot_Processed['Translated Reviews'] = Appbot[['Translated Subject','Translated Body']].apply(lambda x : '{}. {}'.format(x[0],x[1]), axis=1)
    print('Finish processing the Appbot Data!\n')
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

    print('Finish processing the SurveyGizmo Data!\n')
    return SurveyGizmo_Processed

def filter_by_date(df, date_threshold):
    """
    The function remove rows whose date is before the given date threshold
    """
    date = datetime.strptime(date_threshold,'%Y-%m-%d').date() # Convert the given threshold (in string) to date
    filtered_id = df['Date'] >= date # Index of the row to drop 
    print(str(len(df) - sum(filtered_id)) + ' records are before the specified date (' + 
          date.strftime("%B") + ' ' + str(date.day) + ', ' + str(date.year) + '). They will be dropped!\n')

    df_filtered = df[filtered_id]

    return df_filtered

def translate_reviews(df):
    """
    This function scans through each review and translate the non-English reviews
    """
    # Set Credentials
    credentials = service_account.Credentials.from_service_account_file(
    '/Users/ivanzhou/Github/Credentials/GCloud-Translation.json')
    translate_client = translate.Client(credentials=credentials) # Set up the Translate Client
    
    # Get column index: for the convenience of extraction
    original_review_col_id = df.columns.get_loc('Original Reviews') # Get the index of the target column 
    translated_review_col_id = df.columns.get_loc('Translated Reviews') # Get the index of the target column 
    
    # Start Translation
    print('Start to translate: ' + str(len(df)) + ' reviews: ')
    for i in range(len(df)):
        if len(df.iloc[i,translated_review_col_id]) < min(4,len(df.iloc[i,original_review_col_id])): # Detect if the translated review is empty
            orginal_review = df.iloc[i,original_review_col_id] # Extract the original review
            try: # Some reviews may contain non-language contents
                if detect(orginal_review) == 'en': # If the original review is in English
                    df.iloc[i,translated_review_col_id] = orginal_review # Copy the original review - do not waste API usage
                else: # Otherwise, call Google Cloud API for translation
                    df.iloc[i,translated_review_col_id] = translate_client.translate(orginal_review, target_language='en')['translatedText']   
            except:
                df.iloc[i,translated_review_col_id] = 'Error: no language detected!'
        if i % 100 == 0: 
            print(str(i+1) +' reviews have been processed!')
    return df    