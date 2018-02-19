import numpy as np
import pandas as pd
import os, os.path
import glob
import re
from google.cloud import translate
from google.oauth2 import service_account
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import argparse
from langdetect import detect
from datetime import datetime
from matplotlib import pyplot as plt
from collections import Counter
import nltk
from nltk.stem import *
import codecs
import time
import re
import nltk
from nltk.corpus import wordnet
from nltk.tag import PerceptronTagger
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words_file_path = 'Data/stop_words/stop_words.txt'


def read_exist_output(file_path):
    xl = pd.ExcelFile(file_path)
    df = xl.parse(xl.sheet_names[0]).fillna('')
    return df


def remove_duplicate(items, print=False):
    dups = set()
    uniqs = []
    for item in items:
        if item not in dups:
            dups.add(item)
            uniqs.append(item)
    if print:
        print(str(len(items) - len(uniqs)) + ' duplicate items have been removed.')
    return uniqs


def read_words_in_file(file_path):
    myfile = open(file_path, 'r', encoding='utf8')
    words = re.findall(r'\w+', myfile.read())
    return words


def write_to_file(words, file_path):
    myfile = open(file_path, "w")
    for stop_word in words:
        myfile.write("%s\n" % stop_word)
    myfile.close()


def read_all_data(col_names, target_folder_path):
    """
    Function to read through all the datasets in the target folder
    todo: add support to the senario where there are multiple Appbot/SurveyGizmo files in the folder
    """
    df = pd.DataFrame()
    # Read in all the dataset
    file_paths = glob.glob(target_folder_path + '*')
    for file_path in file_paths:
        # We need to ignore the previously generated output file, which contains 'py' in the end of filename
        if file_path.split('.')[-2][-2:] == 'py':  # All of the code-generated file contains 'py' in the end of filename
            os.remove(file_path)  # Remove it - we will create a new one
        else:
            file_format = file_path.split('.')[-1]
            if file_format == 'xlsx':
                xl = pd.ExcelFile(file_path)
                SurveyGizmo_df = xl.parse(xl.sheet_names[0]).fillna('')
                SurveyGizmo_Processed = process_surveygizmo_df(SurveyGizmo_df, col_names)
                df = pd.concat([df, SurveyGizmo_Processed])  # Merged the dataframes
            else:
                Appbot_df = pd.read_csv(file_path).fillna('')
                Appbot_Processed = process_appbot_df(Appbot_df, col_names)
                df = pd.concat([df, Appbot_Processed])  # Merged the dataframes
    return df


def create_empty_df(n, col_names):
    df = pd.DataFrame('', index=range(n), columns=col_names)
    return df


def extract_version_SG(SG_Col):
    """
    Function to extract the version information from the Corresponding Column in SurveyGizmo
    todo: add support to the format of 9.34 - now we can only extract 9.3
    todo: add support to the format of 9.0.1 - now we can only extracvt 9.0
    """
    version_list = []
    for i in range(len(SG_Col)):
        string = SG_Col[i]  # Extract the string in the current row
        locator = string.find("FxiOS/")  # Locate the target term in each string
        if locator > 0:  # Find the keyword
            version_code = string.split("FxiOS/", 1)[1].split(' ')[0]  # Example: 10.0b6373
            version = re.findall("^\d+\.\d+\.\d+|^\d+\.\d+", version_code)[
                0]  # Extract the float number in the string with multiple dot
            digits = version.split('.')
            if len(digits) >= 2:  # 10.1 or 10.0.1
                version = float(digits[0] + '.' + digits[
                    1])  # Temporary use - just capture the first two digits so that we can return as a number
            else:
                version = int(version)
        else:
            version = 0
        version_list.append(version)
        # print('Origin: ' + string + ', Version: ' + str(version))
    return version_list


def process_appbot_df(Appbot, col_names):
    """
    Function to Process the Appbot Dataframe
    """
    Appbot_Processed = create_empty_df(len(Appbot), col_names)  # Initialize a new dataframe

    Appbot_Processed['Store'] = Appbot['Store']
    Appbot_Processed['Source'] = 'Appbot'
    Appbot_Processed['Date'] = pd.to_datetime(Appbot['Date']).dt.date
    Appbot_Processed['Version'] = Appbot['Version']
    Appbot_Processed['Rating'] = Appbot['Rating']
    # Appbot_Processed['Emotion'] =  Appbot['Emotion']
    Appbot_Processed['Original Reviews'] = Appbot[['Subject', 'Body']].apply(lambda x: '{}. {}'.format(x[0], x[1]),
                                                                             axis=1)
    Appbot_Processed['Translated Reviews'] = Appbot[['Translated Subject', 'Translated Body']].apply(
        lambda x: '{}. {}'.format(x[0], x[1]), axis=1)
    print('Finish processing the Appbot Data!\n')
    return Appbot_Processed


def process_surveygizmo_df(SurveyGizmo, col_names):
    """
    Function to Process the SurveyGizmo Dataframe
    """
    SurveyGizmo_Processed = create_empty_df(len(SurveyGizmo), col_names)  # Initialize a new dataframe
    SurveyGizmo_Processed['Store'] = 'iOS'
    SurveyGizmo_Processed['Source'] = 'SurveyGizmo'
    SurveyGizmo_Processed['Date'] = pd.to_datetime(SurveyGizmo[SurveyGizmo.columns[0]]).dt.date
    SurveyGizmo_Processed['Version'] = extract_version_SG(SurveyGizmo['Extended User Agent'])
    # SurveyGizmo_Processed['Emotion'] = SurveyGizmo[SurveyGizmo.columns[5]]
    SurveyGizmo_Processed['Original Reviews'] = SurveyGizmo[[SurveyGizmo.columns[5], SurveyGizmo.columns[6]]].apply(
        lambda x: '{}{}'.format(x[0], x[1]), axis=1)
    SurveyGizmo_Processed['Translated Reviews'] = ''

    print('Finish processing the SurveyGizmo Data!\n')
    return SurveyGizmo_Processed


def filter_by_date(df, date_threshold):
    """
    The function remove rows whose date is before the given date threshold
    """
    date = datetime.strptime(date_threshold, '%Y-%m-%d').date()  # Convert the given threshold (in string) to date
    filtered_id = df['Date'] >= date  # Index of the row to drop
    print(str(len(df) - sum(filtered_id)) + ' records are before the specified date (' +
          date.strftime("%B") + ' ' + str(date.day) + ', ' + str(date.year) + '). They will be dropped!\n')

    df_filtered = df[filtered_id]

    return df_filtered


def filter_by_version(df, version_threshold):
    """
    The function remove rows whose date is before the given date threshold
    """

    filtered_id = df['Version'] >= version_threshold  # Index of the row to drop
    print(str(len(df) - sum(filtered_id)) + ' records are before the specified version (' +
          str(version_threshold) + '). They will be dropped!\n')

    df_filtered = df[filtered_id]

    return df_filtered


def translate_reviews(df):
    """
    This function scans through each review and translate the non-English reviews
    """
    translate_client = translate.Client()
    # Get column index: for the convenience of extraction
    original_review_col_id = df.columns.get_loc('Original Reviews')  # Get the index of the target column
    translated_review_col_id = df.columns.get_loc('Translated Reviews')  # Get the index of the target column

    # Start Translation
    print('Start to translate: ' + str(len(df)) + ' reviews: ')
    for i in range(len(df)):
        if len(df.iloc[i, translated_review_col_id]) < min(4, len(
                df.iloc[i, original_review_col_id])):  # Detect if the translated review is empty
            orginal_review = df.iloc[i, original_review_col_id]  # Extract the original review
            try:  # Some reviews may contain non-language contents
                if detect(orginal_review) == 'en':  # If the original review is in English
                    df.iloc[
                        i, translated_review_col_id] = orginal_review  # Copy the original review - do not waste API usage
                else:  # Otherwise, call Google Cloud API for translation
                    df.iloc[i, translated_review_col_id] = \
                        translate_client.translate(orginal_review, target_language='en')['translatedText']
            except:
                df.iloc[i, translated_review_col_id] = 'Error: no language detected!'
        if i % 100 == 0:
            print(str(i + 1) + ' reviews have been processed!')
    print('All reviews have been processed!')

    return df


def get_counter_contents(counter, sorted=False):
    """
    Get the labels and counts in the counters
    """
    labels, values = zip(*counter.items())
    if sorted:
        ind_sorted = np.array(values).argsort()[::-1]
        labels = np.array(labels)[ind_sorted]
        values = np.array(values)[ind_sorted]
    return labels, values


def plot_hists(values):
    """
    Plot the histogram with the given value. Since the histogram is skewed to the right, I also plot a hist at
    the range [0,9] at a detailed look
    """
    plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))  # Create two subplots side by side

    # A histogram of the entire frequency dataset
    axes[0].hist(values, bins=50)
    axes[0].set_title('Histogram of the frequency of all the words')
    # Set common labels
    axes[0].set_xlabel('Word Frequency')
    axes[0].set_ylabel('# Unique Words')

    # A histogram with a narrow range to get a detailed look at the left-most bin
    axes[1].hist(values, range=(0, 10))
    axes[1].set_title('Histogram of the frequency of words with range [0,9]')
    # Set common labels
    axes[1].set_xlabel('Word Frequency')
    axes[1].set_ylabel('# Unique Words')
    plt.show()


def word_process(word):
    sbStem = SnowballStemmer("english")
    word_processed = sbStem.stem(word)
    word_processed = word_processed.lower()
    return word_processed


def phrase_process(phrase):
    """
    Process words in a phrase individually
    :param phrase: a phrase with multiple words
    :return: a phrase of processed words (string format)
    """
    processed_phrase = ''
    stop_words = get_stop_words()
    for word in re.findall(r'\w+', phrase):
        word_processed = word_process(word)
        if word_processed not in stop_words:
            processed_phrase += ' ' + word_process(word)
    return processed_phrase


def get_stop_words(process_word=True, additional_stop_words=[]):
    stop_words = read_words_in_file(stop_words_file_path)

    if len(additional_stop_words) > 0:
        stop_words += additional_stop_words

    if process_word:
        stop_words = [word_process(word) for word in stop_words]
    stop_words = remove_duplicate(stop_words)
    return stop_words


def clean_words(counter, stop_words):
    words = []
    values = []
    for i in range(len(counter) - 1, -1, -1):
        word = counter[i][0]
        if word in stop_words or len(word) == 1 or word.isdigit() == True:
            del counter[i]
        else:
            words.insert(0, word)
            values.insert(0, counter[i][1])
    return words, values


def plot_bar_plots(X, Y, label_x='', label_y='', title=''):
    fig = plt.figure(figsize=(10, 6))
    width = .35
    indices = np.arange(len(X))
    plt.bar(indices, Y, width=width)
    plt.xticks(indices + width / 2, X)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.show()


def select_from_list_a_based_on_list_b(list_a, list_b, min_thresh, k):
    """
    A function to select from the list A based on the value from the list B
    :param list_a: a list of items, like strings
    :param list_b: a list of int/float
    :param thresh: minimum threshold
    :param k: select top k if none of the values meets the threshold
    :return: a list of items from the list A
    """
    list_a_selected = [a for a, b in zip(list_a, list_b) if b >= min_thresh]
    if len(list_a_selected) <= k:
        list_b_selected = [b for b in list_b if b >= min_thresh]
    else:
        top_k_ranks = np.argsort(np.array(list_b))[-k:]
        list_a_selected = [list_a[i] for i in top_k_ranks]
        list_b_selected = [list_b[i] for i in top_k_ranks]
    return list_a_selected, list_b_selected


def compute_keywords_freq(texts, additional_stop_words=[], process_word=True):
    """
    Function to compute the term frequency of high frequent terms
    """
    stop_words = get_stop_words(process_word=process_word,
                                additional_stop_words=additional_stop_words)
    counter = Counter()
    for text in texts:
        if process_word:
            counter.update([word_process(word)
                            for word in re.findall(r'\w+', text)
                            if word.lower() not in stop_words and len(word) > 2])
        else:
            counter.update([word for word in re.findall(r'\w+', text)
                            if word.lower() not in stop_words and len(word) > 2])
    if len(counter) > 0:
        words, counts = get_counter_contents(counter, sorted=True)
        return words, counts
    else:
        return [], []


def select_keywords_on_freq(texts, k=50, min_thresh=0, process_word=True, additional_stop_words=[], get_counts=False):
    """
    Function to compute the term frequency of high frequent terms
    """
    words, counts = compute_keywords_freq(texts,
                                          additional_stop_words=additional_stop_words,
                                          process_word=process_word)
    words_selected, counts_selected = select_from_list_a_based_on_list_b(words, counts,
                                                                         min_thresh=min_thresh,
                                                                         k=k)
    if get_counts:
        return words_selected, counts_selected
    else:
        return words_selected


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
    annotations = client.analyze_sentiment(document=document)
    score, magnitude, sent_socres, sent_magnitudes = interpret_sentiment(annotations)
    return score, magnitude, sent_socres, sent_magnitudes


def sentiment_analysize(texts, client):
    print('Start Sentiment Analysis: ' + str(len(texts)) + ' reviews: ')
    scores = np.zeros(len(texts))
    magnitudes = np.zeros(len(texts))
    sent_scores_list = []
    sent_magnitudes_list = []
    count = 0
    for i, review in enumerate(texts):
        if (i % 500 == 0 and i > 0):
            print('500 reviews have been processed, and the programme need to be paused for 60 seconds!')
            time.sleep(60)
        scores[i], magnitudes[i], sent_scores, sent_magnitudes = get_sentiment(client, review)
        sent_scores_list.append(sent_scores)
        sent_magnitudes_list.append(sent_magnitudes)
        if i % 100 == 0:
            print(str(i + 1) + ' reviews have been processed!')
    print('All reviews have been processed!')
    return scores, magnitudes, sent_scores_list, sent_magnitudes_list


def discretize_sentiment(score, magnitudes):
    if np.abs(magnitudes) <= 0.25 or np.abs(score) <= 0.15:
        sentiment = 'Neutral'
    elif score > 0:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    return sentiment


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


def identify_keywords(df, n_top_words=30):
    allWords, allWordsFreq = compute_keywords_freq(df['Translated Reviews'])
    topWords = allWords[:n_top_words]
    keywords_list = []
    for review in df['Translated Reviews']:
        counter = Counter([word_process(word) for word in
                           re.findall(r'\w+', review)])  # count all the words and their frequency in the file
        words = []
        for word in topWords:
            if counter[word] > 0:
                words.append(word)
        keywords_list.append(words)
    df['Keywords'] = keywords_list
    return df


def process_words_list(words_list):
    """
    Function to process a list of words and keep record of their original form
    :param words_list: a list of words
    :return: a list of the processed words and a dictionary for recovery
    """
    words_processed_list = []
    words_processed_dict = {}
    for words in words_list:
        if len(words.split(' ')) > 1:  # Phrases
            word_processed = []
            word_list = re.findall(r'\w+', words)
            for word in word_list:
                word_processed.append(word_process(word))
            words_processed_list.append(word_processed)
            words_processed_dict[repr(word_processed)] = words
        else:  # Single word
            words_processed = word_process(words)
            words_processed_list.append(words_processed)
            words_processed_dict[words_processed] = words
        if '/' in words or '&' in words:  # There is a OR condition, record keywords separately
            word_list = re.split('/| & ', words)
            for word in word_list:
                word_processed = word_process(word)
                words_processed_list.append(word_processed)
                words_processed_dict[word_processed] = word

    return words_processed_list, words_processed_dict


def process_texts(texts):
    """
    Function to process review texts
    Output is a list of processed reviews in strings
    """
    texts_processed = []
    for text in texts:
        text_processed = phrase_process(text)
        texts_processed.append(text_processed)
    return texts_processed


def find_word_pair_in_text(text, word1, word2, distance=5):
    """
    The function check if two words are present in the given text within the given distance
    outputs:
    - found: boolean, if the two words are found
    indices as the word pair may present for more than once
    """
    found = False
    for i in range(len(text)):
        if text[i] == word1:  # Find the first word
            for j in range(max(i - distance, 0), min(i + distance, len(text))):
                if text[j] == word2:  # Find the second word in the neighbours
                    found = True
                    return found
    return found


def find_phrase_in_text(text, phrase):
    """
    Phrase is given in a list of words
    """
    found = False
    for i in range(len(phrase) - 1):
        for j in range(i + 1, len(phrase)):
            result = find_word_pair_in_text(text, phrase[i], phrase[j])
            if result:
                found = True
                return found
    return found


def find_words(texts, target_words):
    """
    Function to find a list of target words in a list of texts
    """
    texts_processed = process_texts(texts)
    words_processed, words_processed_dict = process_words_list(target_words)

    words_in_texts = []
    for text in texts_processed:
        text = [word for word in re.findall(r'\w+', text)]  # convert text in string to a list of words
        words_in_text = []
        for words in words_processed:
            if isinstance(words, list):  # Is a Phrase
                found = find_phrase_in_text(text, words)
                if found:
                    words_in_text.append(words_processed_dict[repr(words)])
            else:  # Is a single word
                if words in text:
                    words_output = words_processed_dict[words]
                    if words_output not in words_in_text:
                        words_in_text.append(words_output)
        words_in_text = str(words_in_text).replace('[', '').replace(']', '').replace('\'', '')
        words_in_texts.append(words_in_text)
    return words_in_texts


def preprocess_cate(cate_file_path):
    """
    Function to pre-process the categorization data
    """
    cate_file = pd.read_csv(cate_file_path)
    cate_file = cate_file.iloc[:, 0:2]  # Only look at the first two columns
    cate_file = cate_file.dropna(axis=0, how='all')  # Drop rows with both columns
    component_prior = 'Unknown'  # Initialize
    for i in range(len(cate_file)):
        if pd.isnull(cate_file.iloc[i, 0]):
            cate_file.iloc[i, 0] = component_prior
        else:
            component_prior = cate_file.iloc[i, 0]
    cate_file.to_csv(cate_file_path, index=False)
    return cate_file


def freq_count(df, target):
    """
    Count the # different items in the input df and return a new df with the freq counts
    """
    counter = Counter()
    for content in df[target]:
        counter.update([word for word in content.split(', ')])
    labels, values = get_counter_contents(counter, sorted=True)
    df_output = create_empty_df(len(labels), [target, '# Reviews'])
    df_output[target] = labels
    df_output['# Reviews'] = values
    return df_output


# Noun Phrase Extraction Support Functions
tagger = PerceptronTagger()
pos_tag = tagger.tag
# Extract none phrases
grammar = r"""
    NP: {<DT>? <JJ>* <NN|NNP|NNS|NNPS>} # NP
    P: {<IN>}           # Preposition
    V: {<VB|V.*>}          # Verb
    PP: {<P> <NP>}      # PP -> P NP
    VP: {<JJ>* <V> <NP|PP>+}  # VP -> V (NP|PP)*
    RP: {<RB|RBR|RBS>}
    VR: {<V> <RP>}
"""
chunker = nltk.RegexpParser(grammar)  # Create phrase tree


def leaves(tree, label_type='VP'):
    """Finds leaf nodes of target phrases in a chunk tree."""
    for subtree in tree.subtrees(filter=lambda t: t.label() == label_type):
        yield subtree.leaves()


def get_terms(tree, label_type):
    for leaf in leaves(tree, label_type):
        term = [w for w, t in leaf]
        yield term


def flatten(npTokenList):
    """
    Flatten phrase lists to get tokens for analysis
    """
    finalList = []
    for phrase in npTokenList:
        token = ''
        for word in phrase:
            token += word + ' '
        finalList.append(token.rstrip())
    return finalList


def get_phrases(review, label_type='VP'):
    """
    This function get the pre-defined phrases from the given reviews
    """
    sentences = split_text(review)
    phrases_lists = []
    for sentence in sentences:
        words = re.findall(r'\w+', sentence)
        if len(words) > 0:  # Some sentence may be made up of only '????.'
            phrases_lists.append(flatten([word for word in get_terms(chunker.parse(pos_tag(words)), label_type)]))
    phrases = [phrase for phrase_list in phrases_lists for phrase in phrase_list]  # Flattern the list
    return phrases


def split_text(text):
    """
    Function to split text into seperate sentences
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences


def parse_and_analyze(sentence):
    """for experiment use only"""
    words = re.findall(r'\w+', sentence)
    return pos_tag(words)


def extract_phrases(texts, target_phrase_type):
    """
    Function to extract phrases of the target type
    :param texts: Group of texts in dataframe
    :param target_phrase_type: define the type of the phrases to catch, e.g. VP, NP
    :return: a list of phrases identified
    """
    texts = texts.as_matrix()
    phrases_list = []
    for text in texts:
        phrases = get_phrases(text, target_phrase_type)
        phrases = str(phrases).replace('[', '').replace(']', '').replace('\'', '')
        phrases_list.append(phrases)
    return phrases_list


def vectorize_tfidf(texts):
    vect = TfidfVectorizer(min_df=1)
    tfidf_encoded = vect.fit_transform(texts)
    return tfidf_encoded


def measure_sim_tfidf(texts, viz=False):
    """
    Function to measure the similarity between texts with TF-IDF
    :param texts: list of texts in arrays
    :param viz: control the visualization
    :return: similarity matrix
    """
    tfidf = vectorize_tfidf(texts)
    similarity_matrix = (tfidf * tfidf.T).A
    if viz:
        plot_heatmap(similarity_matrix, title='Heatmap of the Similarity Matrix')
    return similarity_matrix


def plot_heatmap(data, title=''):
    """
    This function plots a heatmap with the input data matrix
    """
    plt.figure(figsize=(10, 5))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.imshow(data, cmap=cmap, interpolation='nearest')  # Create a heatmap
    plt.colorbar()  # Add a Color Bar by the side
    if len(title) > 0:
        plt.title(title)
    plt.show()


def cut_feedbacks(feedbacks, length=2):
    """
    Function to cut long feedbacks into sentences with specified length
    :param feedbacks: feedbacks in dataframe
    :param length: maximum length of a group of sentences
    :return: a list of cut sentences and a dictionary that map each sentence ID to feedback ID
    """
    feedbacks = feedbacks.as_matrix()
    sentence_to_feedback = {}
    sentences_list = []
    for i, text in enumerate(feedbacks):
        sentences = split_text(text)
        if len(sentences) > length:
            for j in range(len(sentences) - (length - 1)):
                sentences_list.append(sentences[j:j + length])
                sentence_to_feedback[len(sentences_list) - 1] = i
        else:
            sentences_list.append(sentences)
            sentence_to_feedback[len(sentences_list) - 1] = i
    # sentences_list = [sentence for sentences in sentences_list for sentence in sentences]  # Flattern the list
    return sentences_list, sentence_to_feedback


def index_df(df):
    indices = np.arange(len(df))
    df['ID'] = indices
    return df


def spam_filter(df, colname='Translated Reviews'):
    spams = np.zeros(len(df))
    for i, row in df.iterrows():
        feedback = row[colname]
        too_long = (len(feedback) > 1000) & (len(re.findall(r'[Ff]irefox|browser', feedback)) < 2)
        emails_match = len(re.findall(r'[\w\.-]+@[\w\.-]+', feedback))
        too_many_digits = len(re.findall(r'\d+', feedback)) > 30
        if (too_long + emails_match + too_many_digits > 0):
            spams[i] = 1
    df['Spam'] = spams
    df_filtered = df[df['Spam'] == 0]
    return df_filtered


def cluster_based_on_similarity(similarity_matrix, thresh):
    """
    Cluster items based on the mutual similarity
    :param similarity_matrix: nxn similarity matrix
    :param thresh: minimum threshold of similarity
    :return: a list of clusters - each cluster will be in a list, even if there is only one item in the cluster
    """
    indice_list = []
    n = similarity_matrix.shape[0]
    for index, x in np.ndenumerate(np.triu(similarity_matrix, k=1)):
        if x >= thresh:
            indice_list.append(list(index))
    clusters = merge_overlapped_lists(indice_list)
    if len(clusters) > 0:
        remaining = list(set(np.arange(n)) - set(np.hstack(clusters)))
        joint_clusters = clusters + remaining
    else:  # No cluster is found
        joint_clusters = np.arange(n)
    clusters_final = [[cluster] if not (isinstance(cluster, list)) else cluster for cluster in joint_clusters]
    return clusters_final


def merge_overlapped_lists(l):
    out = []
    while len(l) > 0:
        first, *rest = l
        first = set(first)

        lf = -1
        while len(first) > lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r))) > 0:
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2

        out.append(list(first))
        l = rest
    return out


def compute_similarity_between_words(word1, word2):
    """
    Compute the semantic similarity between a word pair based on WordNet
    By default, the similarity bewteen 2 words are 0
    """
    wordFromList1 = wordnet.synsets(word1)
    wordFromList2 = wordnet.synsets(word2)
    if wordFromList1 and wordFromList2:
        sim = wordFromList1[0].path_similarity(wordFromList2[0])
        if sim:
            return sim
        else:
            return 0
    else:
        return 0


def compute_similarity_words(words):
    """
    Compute the semantic similarity between word pairs in words
    :param words: a list of words
    :return: a nxn squared matrix for similarity
    """
    n = len(words)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i==j:
                sim = 1
            else:
                sim = compute_similarity_between_words(words[i], words[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    return similarity_matrix
