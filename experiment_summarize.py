from Categorization import *
from Summarize import *
from support_functions import *

def data_processing(col_names, target_folder_path, date_threshold='', version_threshold='', save_output=True):
    """
    Data Processing Pipeline
    :param col_names:
    :param target_folder_path:
    :param date_threshold:
    :param version_threshold:
    :param save_output: trigger to save the output; True by default
    :return:
    """
    """
    df = read_all_data(col_names, target_folder_path)
    if len(date_threshold) > 0:
        df = filter_by_date(df, date_threshold)  # Remove rows whose date is before the given date thershold
    if version_threshold > 0:
        df = filter_by_version(df, version_threshold)  # Remove rows whose version is before the given date thershold
    df = translate_reviews(df)  # Translate non-English reviews
    df = measure_sentiments(df)  # Sentiment Analysis
    df.to_csv('temp.csv')
    df = pd.read_csv('temp.csv')
    df = categorize(df)
    df.to_csv('temp.csv')
    """
    df = pd.read_csv('temp.csv')
    df = summarize(df)

target_folder_path = 'Data/2017_12_28_copy/'
date_threshold = '2017-11-13'
version_threshold = 10
col_names = ['Store','Source','Date','Version','Rating','Original Reviews','Translated Reviews','Sentiment','Components','Features','Keywords']
df = data_processing(col_names,target_folder_path,date_threshold,version_threshold)


