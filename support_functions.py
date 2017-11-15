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

global cate_file_path 
cate_file_path = 'Data/Categorization.csv'

def read_exist_output(file_path):
    xl = pd.ExcelFile(file_path)
    df = xl.parse(xl.sheet_names[0]).fillna('')
    return df

def remove_duplicate(items,print = False):
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
    with codecs.open(file_path,encoding='utf-8', errors='ignore') as myfile:
        words = re.findall(r'\w+', myfile.read())
    return words

def write_to_file(words, file_path):
    myfile = open(stop_word_file, "w")
    for stop_word in stop_words:
        myfile.write("%s\n" % stop_word)
    myfile.close()

def data_processing(col_names, target_folder_path,date_threshold = '', save_csv = True):
    df = read_all_data(col_names,target_folder_path)
    
    if len(date_threshold) > 0:
        df = filter_by_date(df, date_threshold) # Remove rows whose date is before the given date thershold
    df = translate_reviews(df) # Translate non-English reviews
    df = measure_sentiments(df) # Sentiment Analysis
    df = identify_keywords(df)
    df = categorize(df)
    target_folder_path = 'Data/2017_10_16/'
    file_path = target_folder_path + 'output_py.xlsx'
    df = read_exist_output(file_path)
    df_comp_counts = freq_count(df,'Components')
    df_features_counts = freq_count(df,'Features')
    # Save into an output file in the target folder
    
    if save_csv:
        output_path = target_folder_path + 'output_py.xlsx'
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        # df.to_csv(output_path,encoding='utf-8')
        df.to_excel(writer,sheet_name='Main',index= False)
        df_comp_counts.to_excel(writer,sheet_name='Component Count',index= False)
        df_features_counts.to_excel(writer,sheet_name='Feature Count',index= False)
        writer.save()
    return df

def read_all_data(col_names,target_folder_path):
    """
    Function to read through all the datasets in the target folder
    todo: add support to the senario where there are multiple Appbot/SurveyGizmo files in the folder
    """
    df = pd.DataFrame()
    # Read in all the dataset
    file_paths = glob.glob(target_folder_path + '*')
    for file_path in file_paths:
        # We need to ignore the previously generated output file, which contains 'py' in the end of filename
        if file_path.split('.')[-2][-2:] == 'py': # All of the code-generated file contains 'py' in the end of filename
            os.remove(file_path) # Remove it - we will create a new one
        else:
            file_format = file_path.split('.')[-1]
            if file_format == 'xlsx':
                xl = pd.ExcelFile(file_path)
                SurveyGizmo_df = xl.parse(xl.sheet_names[0]).fillna('')
                SurveyGizmo_Processed = process_surveygizmo_df(SurveyGizmo_df,col_names)
                df = pd.concat([df,SurveyGizmo_Processed]) # Merged the dataframes
            else:
                Appbot_df = pd.read_csv(file_path).fillna('')
                Appbot_Processed = process_appbot_df(Appbot_df,col_names)
                df = pd.concat([df,Appbot_Processed]) # Merged the dataframes
    return df

def create_empty_df(n,col_names):
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
        string = SG_Col[i] # Extract the string in the current row
        locator = string.find("FxiOS/") # Locate the target term in each string
        if locator > 0: # Find the keyword
            version_code = string.split("FxiOS/",1)[1].split(' ')[0]  # Example: 10.0b6373
            version = re.findall("^\d+\.\d+\.\d+|^\d+\.\d+", version_code)[0] # Extract the float number in the string
        else:
            version = ''
        version_list.append(version)
    return version_list

def process_appbot_df(Appbot,col_names):
    """
    Function to Process the Appbot Dataframe
    """
    Appbot_Processed = create_empty_df(len(Appbot),col_names) # Initialize a new dataframe
    
    Appbot_Processed['Store'] =  Appbot['Store']
    Appbot_Processed['Source'] =  'Appbot'
    Appbot_Processed['Date'] =  pd.to_datetime(Appbot['Date']).dt.date
    Appbot_Processed['Version'] =  Appbot['Version']
    Appbot_Processed['Rating'] =  Appbot['Rating']
    #Appbot_Processed['Emotion'] =  Appbot['Emotion']
    Appbot_Processed['Original Reviews'] = Appbot[['Subject','Body']].apply(lambda x : '{}. {}'.format(x[0],x[1]), axis=1)
    Appbot_Processed['Translated Reviews'] = Appbot[['Translated Subject','Translated Body']].apply(lambda x : '{}. {}'.format(x[0],x[1]), axis=1)
    print('Finish processing the Appbot Data!\n')
    return Appbot_Processed

def process_surveygizmo_df(SurveyGizmo,col_names):
    """
    Function to Process the SurveyGizmo Dataframe
    """
    SurveyGizmo_Processed = create_empty_df(len(SurveyGizmo),col_names) # Initialize a new dataframe
    
    SurveyGizmo_Processed['Store'] = 'iOS'
    SurveyGizmo_Processed['Source'] = 'Browser'
    SurveyGizmo_Processed['Date'] = pd.to_datetime(SurveyGizmo[SurveyGizmo.columns[0]]).dt.date
    SurveyGizmo_Processed['Version'] = extract_version_SG(SurveyGizmo[SurveyGizmo.columns[3]])
    #SurveyGizmo_Processed['Emotion'] = SurveyGizmo[SurveyGizmo.columns[5]]
    SurveyGizmo_Processed['Original Reviews'] = SurveyGizmo[[SurveyGizmo.columns[6],SurveyGizmo.columns[7]]].apply(lambda x : '{}{}'.format(x[0],x[1]), axis=1)
    SurveyGizmo_Processed['Translated Reviews'] = ''

    print('Finish processing the SurveyGizmo Data!\n')
    return SurveyGizmo_Processed

def filter_by_date(df, date_threshold):
    """
    The function remove rows whose date is before the given date threshold
    """
    date = datetime.strptime(date_threshold,'%Y-%m-%d').date() # Convert the given threshold (in string) to date
    filtered_id = df['Date'] >= date # Index of the row to drop 
    print(str(len(df) - sum(filtered_id)) + ' records are before the specified date (' + 
          date.strftime("%B") + ' ' + str(date.day) + ', ' + str(date.year) + '). They will be dropped!\n')

    df_filtered = df[filtered_id]

    return df_filtered

def translate_reviews(df):
    """
    This function scans through each review and translate the non-English reviews
    """
    translate_client = translate.Client()
    # Get column index: for the convenience of extraction
    original_review_col_id = df.columns.get_loc('Original Reviews') # Get the index of the target column 
    translated_review_col_id = df.columns.get_loc('Translated Reviews') # Get the index of the target column 
    
    # Start Translation
    print('Start to translate: ' + str(len(df)) + ' reviews: ')
    for i in range(len(df)):
        if len(df.iloc[i,translated_review_col_id]) < min(4,len(df.iloc[i,original_review_col_id])): # Detect if the translated review is empty
            orginal_review = df.iloc[i,original_review_col_id] # Extract the original review
            try: # Some reviews may contain non-language contents
                if detect(orginal_review) == 'en': # If the original review is in English
                    df.iloc[i,translated_review_col_id] = orginal_review # Copy the original review - do not waste API usage
                else: # Otherwise, call Google Cloud API for translation
                    df.iloc[i,translated_review_col_id] = translate_client.translate(orginal_review, target_language='en')['translatedText']   
            except:
                df.iloc[i,translated_review_col_id] = 'Error: no language detected!'
        if i % 100 == 0: 
            print(str(i+1) +' reviews have been processed!')
    print('All reviews have been processed!')
    return df    

def get_counter_contents(counter,sorted = False):
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
    fig, axes = plt.subplots(1, 2,figsize=(15,8)) # Create two subplots side by side

    # A histogram of the entire frequency dataset
    axes[0].hist(values, bins=50)
    axes[0].set_title('Histogram of the frequency of all the words')
    # Set common labels
    axes[0].set_xlabel('Word Frequency')
    axes[0].set_ylabel('# Unique Words')

    # A histogram with a narrow range to get a detailed look at the left-most bin
    axes[1].hist(values, range=(0,10))
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

def get_stop_words(file_path):
    stop_words = [word_process(word) for word in read_words_in_file(file_path)]
    stop_words = remove_duplicate(stop_words)
    return stop_words

def clean_words(counter, stop_words):
    words = []
    values = []
    for i in range(len(counter)-1,-1,-1):
        word = counter[i][0]
        if word in stop_words or len(word) == 1 or word.isdigit() == True:
            del counter[i]
        else:
            words.insert(0,word)
            values.insert(0,counter[i][1])
    return words, values

def plot_bar_plots(X, Y,label_x='', label_y='',title=''):
    fig = plt.figure(figsize=(10, 6))
    width = .35
    indices = np.arange(len(X))
    plt.bar(indices, Y, width = width)
    plt.xticks(indices+width/2, X)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.show()
    
def compute_keywords_freq(reviews, stop_words_file_path = 'Data/stop_words/stop_words.txt',viz = False,n_top_words = 15):
    counter = Counter()
    for review in reviews:
        counter.update([word_process(word) for word in re.findall(r'\w+', review)])
    counter = counter.most_common()
    stop_words = get_stop_words(stop_words_file_path)
    words, freq = clean_words(counter, stop_words)
    if viz:
        plot_bar_plots(words[:n_top_words], freq[:n_top_words],'High Frequent Keywords','Frequency','Words with the Top Frequency')
    return words, freq

def interpret_sentiment(annotations):
    score = annotations.document_sentiment.score 
    magnitude = annotations.document_sentiment.magnitude/len(annotations.sentences) # Take the average
    sent_scores = []
    sent_magnitudes = []
    for index, sentence in enumerate(annotations.sentences):
        sent_scores.append(sentence.sentiment.score)
        sent_magnitudes.append(sentence.sentiment.magnitude)
    return score, magnitude, sent_scores, sent_magnitudes

def get_sentiment(client,content):
    document = types.Document(
        content=content,
        type=enums.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(document=document)
    score, magnitude, sent_socres, sent_magnitudes = interpret_sentiment(annotations)
    return score, magnitude, sent_socres, sent_magnitudes

def sentiment_analysize(texts,client):
    print('Start Sentiment Analysis: ' + str(len(texts)) + ' reviews: ')
    scores = np.zeros(len(texts))
    magnitudes = np.zeros(len(texts))
    sent_scores_list = []
    sent_magnitudes_list = []
    count = 0
    for i, review in enumerate(texts):
        if i == 500:
            print('500 reviews have been processed, and the programme need to be paused for 60 seconds!')
            time.sleep(60)
        scores[i], magnitudes[i], sent_scores, sent_magnitudes = get_sentiment(client,review)
        sent_scores_list.append(sent_scores)
        sent_magnitudes_list.append(sent_magnitudes)
        if i%100 == 0:
            print(str(i+1) + ' reviews have been processed!')
    print('All reviews have been processed!')
    return scores, magnitudes, sent_scores_list, sent_magnitudes_list

def discretize_sentiment(score, magnitudes):
    if np.abs(magnitudes) <= 0.25 or np.abs(score)  <= 0.15 :
        sentiment = 'Neutral'
    elif score > 0:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    return sentiment

def measure_sentiments(df):
    client = language.LanguageServiceClient()
    scores, magnitudes, sent_scores_list, sent_magnitudes_list = sentiment_analysize(df['Translated Reviews'],client)
    #df['sentiment_score'] = scores
    #df['sentiment_magnitude'] = magnitudes
    #df['score_by_sentence'] = sent_scores_list
    #df['magnitude_by_sentence'] = sent_magnitudes_list
    
    sentiments = []
    for i in range(len(df)):
        sentiment = discretize_sentiment(scores[i], magnitudes[i])
        sentiments.append(sentiment)
    df['Sentiment'] = sentiments
    return df

def identify_keywords(df,n_top_words=30):
    allWords, allWordsFreq = compute_keywords_freq(df['Translated Reviews'],viz = False)
    topWords = allWords[:n_top_words]
    keywords_list = []
    for review in df['Translated Reviews']:
        counter = Counter([word_process(word) for word in re.findall(r'\w+', review)]) # count all the words and their frequency in the file
        words = []
        for word in topWords:
            if counter[word] > 0:
                words.append(word)
        keywords_list.append(words)
    df['Keywords'] = keywords_list
    return df

def process_words_list(words_list):
    words_processed_list = []
    words_processed_dict = {}
    for words in words_list:
        if len(words.split(' '))>1: # Phrases
            word_processed = []
            word_list = re.findall(r'\w+', words)
            for word in word_list:
                word_processed.append(word_process(word))
            words_processed_list.append(word_processed)
            words_processed_dict[repr(word_processed)] = words
        else: # Single word
            words_processed = word_process(words)
            words_processed_list.append(words_processed)
            words_processed_dict[words_processed] = words
        if '/' in words or '&' in words: # There is a OR condition, record keywords separately
            word_list = re.split('/| & ',words)
            for word in word_list:
                word_processed = word_process(word)
                words_processed_list.append(word_processed)
                words_processed_dict[word_processed] = words
        
    return words_processed_list, words_processed_dict

def process_texts(texts):
    """
    Function to process review texts
    Output is a list of processed reviews
    """
    texts_processed = []
    for text in texts:
        text_processed = []
        words = re.findall(r'\w+', text)
        for word in words:
            text_processed.append(word_process(word))
        texts_processed.append(text_processed)
    return texts_processed

def find_word_pair_in_text(text, word1,word2,distance = 5):
    """
    The function check if two words are present in the given text within the given distance
    outputs:
    - found: boolean, if the two words are found
    indices as the word pair may present for more than once
    """
    found = False
    for i in range(len(text)):
        if text[i] == word1: # Find the first word
            for j in range(max(i-distance,0),min(i+distance,len(text))): 
                if text[j] == word2: # Find the second word in the neighbours
                    found = True
                    return found
    return found

def find_phrase_in_text(text,phrase):
    """
    Phrase is given in a list of words
    """
    found = False
    for i in range(len(phrase)-1):
        for j in range(i+1,len(phrase)):
            result = find_word_pair_in_text(text, phrase[i],phrase[j])
            if result:
                found = True
                return found
    return found

def find_words(texts,target_words):
    """
    Function to find a list of target words in a list of texts
    """
    texts_processed = process_texts(texts)
    words_processed, words_processed_dict = process_words_list(target_words)

    words_in_texts = []
    for text in texts_processed:
        words_in_text = []
        for words in words_processed:
            if isinstance(words, list): # Is a Phrase
                found = find_phrase_in_text(text,words)
                if found:
                    words_in_text.append(words_processed_dict[repr(words)])
            else: # Is a single word
                if words in text:
                    words_output = words_processed_dict[words]
                    if words_output not in words_in_text:
                        words_in_text.append(words_output)
        words_in_text = str(words_in_text).replace('[','').replace(']','').replace('\'','')
        words_in_texts.append(words_in_text)  
    return words_in_texts

def preprocess_cate(cate_file_path):
    """
    Function to pre-process the categorization data
    """
    cate_file = pd.read_csv(cate_file_path)
    cate_file = cate_file.iloc[:,0:2] # Only look at the first two columns
    cate_file = cate_file.dropna(axis=0, how='all') # Drop rows with both columns
    component_prior = 'Unknown' # Initialize
    for i in range(len(cate_file)):
        if pd.isnull(cate_file.iloc[i,0]):
            cate_file.iloc[i,0] = component_prior
        else:
            component_prior = cate_file.iloc[i,0]
    cate_file.to_csv(cate_file_path, index=False)
    return cate_file

def categorize(df):
    cate_file = pd.read_csv(cate_file_path)
    components = list(cate_file['Components'].unique())
    features = list(cate_file['Feature'].unique())
    print('Start to categorize: ' + str(len(df)) + ' reviews: ')
    texts = df['Translated Reviews']
    # Find the pre-defined components and features in the texts
    components_found = find_words(texts,components)
    features_found = find_words(texts,features)

    # If there is a text that has no component but has features, add "Firefox" as the component
    for i in range(len(texts)):
        if len(components_found[i]) == 0:
            if len(features_found[i]) >0: # General Browser Issue
                components_found[i] = 'Firefox/Browser/App'
            else:
                components_found[i] = 'Others'


    # Identify keywords (excluding stop words and components)
    keywords,freq = compute_keywords_freq(texts)
    for component in components:
        for word in re.findall(r'\w+', component) :
            word_processed = word_process(word)
            if word_processed in keywords:
                keywords.remove(word_processed)
    top_keywords = keywords[:30]
    keywords_found = find_words(texts,top_keywords)
    df['Components'] = components_found
    df['Features'] = features_found
    df['Keywords'] = keywords_found
    return df

def freq_count(df,target):
    """
    Count the # different items in the input df and return a new df with the freq counts
    """
    counter = Counter()
    for content in df[target]:
        counter.update([word for word in content.split(', ')])
    labels, values = get_counter_contents(counter,sorted = True)
    df_output=create_empty_df(len(labels),[target,'# Reviews'])
    df_output[target] = labels
    df_output['# Reviews'] = values
    return df_output