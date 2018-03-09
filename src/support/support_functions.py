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
from collections import Counter
import nltk
from nltk.stem import *
import codecs
import time
import re
import nltk
from nltk.corpus import wordnet
from nltk.tag import PerceptronTagger
import scipy.stats as ss
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words_file_path = 'spec/stop_words/stop_words.txt'


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


def clean_text_after_translation(text):
    text.replace("&#39;", "\s")
    return text


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
    if isNaN(phrase):
         return ''
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


def find_word_pair_in_text(text, word1, word2, distance=2):
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
    n = len(phrase)
    for i in range(n-1):
        result = find_word_pair_in_text(text, phrase[i], phrase[i+1])
        if not result:  # there is one combo that is not found
            return False
    return True


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
    if isNaN(review):
         return []
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


def measure_sim_tfidf(texts):
    """
    Function to measure the similarity between texts with TF-IDF
    :param texts: list of texts in arrays
    :param viz: control the visualization
    :return: similarity matrix
    """
    tfidf = vectorize_tfidf(texts)
    similarity_matrix = (tfidf * tfidf.T).A
    return similarity_matrix


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




def cluster_based_on_similarity(similarity_matrix, sim_thresh=0.3, size_thresh=4):
    """
    Cluster items based on the mutual similarity
    :param similarity_matrix: nxn similarity matrix
    :param thresh: minimum threshold of similarity
    :return: a list of clusters - each cluster will be in a list, even if there is only one item in the cluster
    """
    indice_list = []
    n = similarity_matrix.shape[0]
    for index, x in np.ndenumerate(np.triu(similarity_matrix, k=1)):
        if x >= sim_thresh:
            indice_list.append(list(index))
    clusters = merge_overlapped_lists(indice_list)
    if len(clusters) > 0:
        remaining = list(set(np.arange(n)) - set(np.hstack(clusters)))
        joint_clusters = clusters + remaining
    else:  # No cluster is found
        joint_clusters = np.arange(n)

    clusters_final = []
    for cluster in joint_clusters:
        if not isinstance(cluster, list):
            cluster = [cluster]

        if len(cluster) >= size_thresh:
            clusters_final.append(cluster)
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


def isNaN(num):
    return num != num


def split_input_words(word_string):
    """
    Split a given string of words; especially process the unpredictable case of "," (without space) and ", " (with space)
    :param word_string: a string of words divided by "," (without space) or ", " (with space)
    :return: a list of words
    """
    words = word_string.split(',')
    output_word_list = []
    for word in words:
        if len(word)> 0:
            word = word[1:] if word[0] == ' ' else word
            word = word[:-1] if word[-1] == ' ' else word
            output_word_list.append(word)
    return output_word_list


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
