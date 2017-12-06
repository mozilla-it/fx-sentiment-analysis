from Categorization import *
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
    df = read_all_data(col_names, target_folder_path)

    if len(date_threshold) > 0:
        df = filter_by_date(df, date_threshold)  # Remove rows whose date is before the given date thershold
    if version_threshold > 0:
        df = filter_by_version(df, version_threshold)  # Remove rows whose version is before the given date thershold
    df = translate_reviews(df)  # Translate non-English reviews
    df = measure_sentiments(df)  # Sentiment Analysis
    df = identify_keywords(df)
    df = categorize(df)
    df_features_counts = freq_count(df, 'Features')
    df_comp_counts = freq_count(df, 'Components')
    df_actions_counts = freq_count(df, 'Actions')

    # Save into an output file in the target folder

    if save_output:
        output_path = target_folder_path + 'output_py.xlsx'
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        # df.to_csv(output_path,encoding='utf-8')
        df.to_excel(writer, sheet_name='Full', index=False)
        df_features_counts.to_excel(writer, sheet_name='Feature Count', index=False)
        df_comp_counts.to_excel(writer, sheet_name='Component Count', index=False)
        df_actions_counts.to_excel(writer, sheet_name='Action Count', index=False)

        writer.save()
        print('Output has been saved to: ' + target_folder_path)
    return df


target_folder_path = 'Data/2017_11_30/'
date_threshold = '2017-11-01'
version_threshold = 10
col_names = ['Store','Source','Date','Version','Rating','Original Reviews','Translated Reviews','Sentiment','Components','Features','Keywords']
df = data_processing(col_names,target_folder_path,date_threshold,version_threshold)


