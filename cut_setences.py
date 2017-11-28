from support_functions import *
df = pd.read_csv('temp.csv')
df = df.replace(np.nan, '', regex=True)

def cut_feedbacks(feedbacks, length = 2):
    """
    Function to cut long feedbacks into sentences with specified length
    :param feedbacks: feedbacks in dataframe
    :param length: maximum length of a group of sentences
    :return: a list of cut sentences and a dictionary that map each sentence ID to feedback ID
    """
    feedbacks = df['Translated Reviews'].as_matrix()
    sentence_to_feedback = {}
    sentences_list = []
    for i, text in enumerate(feedbacks):
        sentences = split_text(text)
        if len(sentences) > length:
            for j in range(len(sentences) - (length-1)):
                sentences_list.append(sentences[j:j+(length-1)])
                sentence_to_feedback[len(sentences_list) - 1] = i
        else:
            sentences_list.append(sentences)
            sentence_to_feedback[len(sentences_list) - 1] = i
    return sentences_list, sentence_to_feedback
