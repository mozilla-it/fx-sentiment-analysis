from src.input.read_data import read_all_data
from src.categorization import *
from src.clustering import *
from src.support_functions import *


def data_processing():
    """
    Data Processing Pipeline
    :param col_names:
    :param target_folder_path:
    :param date_threshold:
    :param version_threshold:
    :param save_output: trigger to save the output; True by default
    :return:
    """
    target_folder_path = 'Input/'
    df = read_all_data()
    df = spam_filter(df, colname='Original Reviews')
    df = translate_reviews(df)  # Translate non-English reviews
    df = spam_filter(df)
    df = measure_sentiments(df)  # Sentiment Analysis
    df['ID'] = np.arange(len(df))  # Add ID Column
    df_categorization, df = categorize(df)
    df.to_csv('temp.csv', index=False)
    df_categorization.to_csv('temp2.csv')
    df_categorization, df_key_issue = cluster_and_summarize(df, df_categorization)
    output_path = target_folder_path + 'output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df.to_csv(output_path + 'feedbacks.csv', index=False)
    df_categorization.to_csv(output_path + 'categorization.csv', index=False)
    df_key_issue.to_csv(output_path + 'key_issue.csv', index=False)
    print('Output has been saved to: ' + output_path)


if __name__ == '__main__':
    data_processing()
