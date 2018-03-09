from src.support.support_functions import *


def filter_by_date(df_feedbacks, df_categorization, df_key_issues, date_threshold):
    """
    The function remove rows whose date is before the given date threshold
    """
    date = datetime.strptime(date_threshold, '%Y-%m-%d').date()  # Convert the given threshold (in string) to date
    df_feedbacks['Date'] = pd.to_datetime(df_feedbacks['Date']).dt.date
    id_list_to_be_removed = list(df_feedbacks[df_feedbacks['Date'] <= date]['ID'])
    if len(id_list_to_be_removed):
        df_feedbacks = df_feedbacks[df_feedbacks['Date'] > date]
        df_categorization = df_categorization[~df_categorization['ID'].isin(id_list_to_be_removed)]
        df_key_issues = df_key_issues[~df_key_issues['ID'].isin(id_list_to_be_removed)]
        df_feedbacks = df_feedbacks.reset_index(drop=True)  # Reset the index as we have removed contents
        df_categorization = df_categorization.reset_index(drop=True)
        df_key_issues = df_key_issues.reset_index(drop=True)
    return df_feedbacks, df_categorization, df_key_issues