from src.support.support_functions import *
from src.pre_processing.Translation import translate_reviews
from src.pre_processing.Sentiment import measure_sentiments


def preprocess(df):
    df = spam_filter(df, colname='Original Reviews')
    df = translate_reviews(df)  # Translate non-English reviews
    df = measure_sentiments(df)  # Sentiment Analysis
    df['ID'] = np.arange(len(df))  # Add ID Column
    df = spam_filter(df)
    return df


def spam_filter(df, colname='Translated Reviews'):
    """
    Function to filter out spam and remove sensitive privacy-related content in feedbacks
    :param df:
    :param colname:
    :return:
    """
    email_regex = '[\w\.-]+@[\w\.-]+'
    phone_regex = '(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'

    def identify_sensitive_info(text):
        return (len(re.findall(email_regex, text)) + len(re.findall(phone_regex, text))) > 0

    def remove_sensitive_info(text):
        """Function to remove sensitive email from feedbacks"""
        text = re.sub(email_regex, "", text)
        text = re.sub(phone_regex, "", text)
        return text

    def spam_detector(text):
        """Function to detect if there is a clue of spam in the sentence"""
        if text == 'Error: no language detected!':
            return 1, text
        if isNaN(text):
            return 1, text
        result = 0  # assume not a spam
        too_long = (len(text) > 1000) & (len(re.findall(r'[Ff]irefox|browser', text)) < 2)
        include_sensitive_info = identify_sensitive_info(text)
        too_many_digits = len(re.findall(r'\d+', text)) > 30
        if include_sensitive_info:
            text = remove_sensitive_info(text)
            result, text = spam_detector(text)  # recursor
        if too_long + too_many_digits + result > 0:
            result = 1  # Spam
        else:
            result = 0
        return result, text

    feedbacks = list(df[colname])
    spams = np.zeros(len(feedbacks))
    need_another_round = True
    while need_another_round:
        need_another_round = False
        feedbacks_processed = []
        for i, feedback in enumerate(feedbacks):
            result, feedback = spam_detector(feedback)
            spams[i] = result
            if result == 0:  #not spam
                feedbacks_processed.append(feedback)
        feedbacks = feedbacks_processed
    df['Spam'] = spams
    df_filtered = df[df['Spam'] == 0]
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered[colname] = feedbacks
    return df_filtered
