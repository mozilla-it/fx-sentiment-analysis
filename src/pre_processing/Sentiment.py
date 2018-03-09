from src.support.support_functions import *


def measure_sentiments(df):
    client = language.LanguageServiceClient()
    scores, magnitudes, sent_scores_list, sent_magnitudes_list = sentiment_analysize(df['Translated Reviews'], client)
    # df['sentiment_score'] = scores
    # df['sentiment_magnitude'] = magnitudes
    # df['score_by_sentence'] = sent_scores_list
    # df['magnitude_by_sentence'] = sent_magnitudes_list

    sentiments = []
    for i in range(len(df)):
        sentiment = discretize_sentiment(scores[i], magnitudes[i])
        sentiments.append(sentiment)
    df['Sentiment'] = sentiments
    return df


def sentiment_analysize(texts, client):
    scores = np.zeros(len(texts))
    magnitudes = np.zeros(len(texts))
    sent_scores_list = []
    sent_magnitudes_list = []
    count = 0
    for i, review in enumerate(texts):
        if (i % 500 == 0 and i > 0):
            time.sleep(60)
        scores[i], magnitudes[i], sent_scores, sent_magnitudes = get_sentiment(client, review)
        sent_scores_list.append(sent_scores)
        sent_magnitudes_list.append(sent_magnitudes)
    return scores, magnitudes, sent_scores_list, sent_magnitudes_list


def discretize_sentiment(score, magnitudes):
    if np.abs(magnitudes) <= 0.25 or np.abs(score) <= 0.15:
        sentiment = 'Neutral'
    elif score > 0:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    return sentiment


def interpret_sentiment(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude / max(1, len(annotations.sentences))  # Take the average
    sent_scores = []
    sent_magnitudes = []
    for index, sentence in enumerate(annotations.sentences):
        sent_scores.append(sentence.sentiment.score)
        sent_magnitudes.append(sentence.sentiment.magnitude)
    return score, magnitude, sent_scores, sent_magnitudes


def get_sentiment(client, content):
    document = types.Document(
        content=content,
        type=enums.Document.Type.PLAIN_TEXT)
    try:
        annotations = client.analyze_sentiment(document=document)
        score, magnitude, sent_socres, sent_magnitudes = interpret_sentiment(annotations)
    except:
        return 0, 0, 0, 0
    return score, magnitude, sent_socres, sent_magnitudes