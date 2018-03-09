from src.input.read_data import read_all_data
from src.pre_processing.preprocessing import preprocess
from src.categorization.categorization import categorize
from src.key_issues.summarization import cluster_and_summarize
from src.data_ouptut.sqlite import load_into_database


def data_processing():
    df = read_all_data()
    df = preprocess(df)
    df_categorization, df = categorize(df)
    df_key_issue = cluster_and_summarize(df, df_categorization)
    load_into_database(df, df_categorization, df_key_issue)


if __name__ == '__main__':
    data_processing()
