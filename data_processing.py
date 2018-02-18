from categorization import *
from clustering import *
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
    df = spam_filter(df)
    df = translate_reviews(df)  # Translate non-English reviews
    df = measure_sentiments(df)  # Sentiment Analysis
    df['ID'] = np.arange(len(df))
    df.to_csv('temp.csv', index=False)
    """
    df = pd.read_csv('temp.csv')
    """
    df_categorization = categorize(df)
    df_categorization.to_csv('temp2.csv')
    """
    df_categorization = pd.read_csv('temp2.csv')
    df_categorization = cluster_and_summarize(df, df_categorization)

    output_path = target_folder_path + 'output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df.to_csv(output_path + 'feedbacks.csv', index=False)
    df_categorization.to_csv(output_path + 'categorization.csv', index=False)
    print('Output has been saved to: ' + output_path)
    return df, df_categorization


target_folder_path = 'Data/2017_12_28/'
date_threshold = '2017-11-13'
version_threshold = 10
col_names = ['Store','Source','Date','Version','Rating','Original Reviews','Translated Reviews','Sentiment']
df, df_categorization = data_processing(col_names,target_folder_path,date_threshold,version_threshold)
