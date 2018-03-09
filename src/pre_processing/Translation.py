from src.support.support_functions import *


def translate_reviews(df):
    """
    This function scans through each review and translate the non-English reviews
    """
    translate_client = translate.Client()
    translated_reviews = []
    for i, row in df.iterrows():
        original_review = row['Original Reviews']
        try:  # Some reviews may contain non-language contents
            translated_review = \
                translate_client.translate(original_review, target_language='en')['translatedText']
        except:
            translated_review = 'Error: no language detected!'
        translated_reviews.append(translated_review)
    df['Translated Review'] = translated_reviews
    return df
