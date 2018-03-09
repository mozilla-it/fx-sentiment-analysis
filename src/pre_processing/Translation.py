from src.support.support_functions import *


def translate_reviews(df):
    """
    This function scans through each review and translate the non-English reviews
    """
    translate_client = translate.Client()
    for i, row in df.iterrows():
        original_review = row['Original Reviews']
        try:  # Some reviews may contain non-language contents
            row['Translated Reviews'] = \
                translate_client.translate(original_review, target_language='en')['translatedText']
        except:
            row['Translated Reviews'] = 'Error: no language detected!'

    return df
