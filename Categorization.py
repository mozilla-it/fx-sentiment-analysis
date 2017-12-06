from support_functions import *

global cate_file_path
cate_file_path = 'Data/Categorization.csv'

def categorize(df):
    cateDict = read_categorization_file(cate_file_path) # Read the content from the categorization file
    print('Start to categorize: ' + str(len(df)) + ' reviews: ')

    #Identify the target phrases
    df['Verb Phrases'] = extract_phrases(df['Translated Reviews'],'VP')
    df['Noun Phrases'] = extract_phrases(df['Translated Reviews'],'NP')

    # Identify high frequent keywords in the given text to supplement the specified categories
    # keywordsFreq = compute_keywords_freq(df['Noun Phrases'], k =80)

    # Find the pre-defined keywords and high frequency keywords in the texts
    keywords_found_VP_list = find_words(df['Verb Phrases'], cateDict.keywords)
    keywords_found_NP_list = find_words(df['Noun Phrases'], cateDict.keywords)
    keywords_found_text_list = find_words(df['Translated Reviews'], cateDict.keywords)
    # keywordsFreq_found_list = find_words(df['Noun Phrases'], keywordsFreq)

    # Map keywords with the corresponding categories
    features_found_list = []
    components_found_list = []
    keywords_found_list = []
    actions_found_list = []
    for i, row in df.iterrows():
        if len(keywords_found_text_list[i]) == 0: # No keywords can be identified in the entire text
            components_found = 'Others'
            features_found = 'Others'
            keywords_found = ''
            actions_found = ''
        else:
            if len(keywords_found_VP_list[i])  > 0:
                keywords_found = keywords_found_VP_list[i]
            elif len(keywords_found_NP_list[i])  > 0:
                keywords_found = keywords_found_NP_list[i]
            else:
                keywords_found = keywords_found_text_list[i]

            components_found = [cateDict.keywords2components[keyword] for keyword in keywords_found.split(', ')]
            features_found = [cateDict.components2features[cateDict.keywords2components[keyword]] for keyword in keywords_found.split(', ')]

            actions_found = []
            for keyword in keywords_found.split(', '):
                if len(keywords_found_VP_list[i]) > 0:
                    action_found = [VP for VP in str(row['Verb Phrases']).split(', ') if word_process(keyword) in phrase_process(VP)]
                else:
                    action_found = extract_phrases_with_keywords(str(row['Translated Reviews']), keyword)
                actions_found.append(action_found)
            components_found = str(components_found).replace('[', '').replace(']', '')
            components_found = list(set(components_found))
            features_found = str(features_found).replace('[', '').replace(']', '')
            components_found = list(set(features_found))
            keywords_found = str(keywords_found).replace('[', '').replace(']', '')
            actions_found = str(actions_found).replace('[', '').replace(']', '')
        keywords_found_list.append(keywords_found)
        features_found_list.append(features_found)
        components_found_list.append(components_found)
        actions_found_list.append(actions_found)

    df['Features'] = features_found_list
    df['Components'] = components_found_list
    df['Actions'] = actions_found_list
    df['Keywords'] = keywords_found_list
    # df['Keywords_Freq'] = keywordsFreq_found_list
    df = df.replace(np.nan, '', regex=True)
    return df


class CategorizationDict:
    def __init__(self, features, components, keywords, components2tiers, components2features, keywords2components):
        self.features = features  # Dictionary that maps components to their features
        self.components = components
        self.keywords = keywords  # List of Keywords
        self.components2tiers = components2tiers  # Dictionary that maps components to their tiers
        self.components2features = components2features # Dictionary that maps components to their features
        self.keywords2components = keywords2components # Dictionary that maps keywords to components

def read_categorization_file(cate_file_path):
    """
    Function to read content from the categorization file
    :param cate_file_path: path to the categorization file
    :return: cate: an instance of categorization that stores all the pre-defined content
    """
    df = pd.read_csv(cate_file_path)

    # Ensure there are no duplicate in components
    assert sum(df['Component'].duplicated()) == 0, \
        "There are duplicated entry in the column Components in the file: %r" % cate_file_path

    components = df['Component'].unique()  # Read components
    features = df['Feature'].unique()

    keywords = []
    components2tiers = {}
    components2features = {}
    keywords2components = {}

    for i, row in df.iterrows():
        component = row['Component']
        components2tiers[component] = row['Tier']
        components2features[component] = row['Feature']
        for keyword in row['Keywords'].split(', '):
            keywords.append(keyword)
            keywords2components[keyword] = component
    cateDict = CategorizationDict(features, components, keywords, components2tiers, components2features, keywords2components)
    return cateDict

def extract_phrases_with_keywords(text, keyword):
    """
    Function to extract a list of phrases within a sentence that include the keyword
    If we find locate the keyword within the sentence, we extract the phrase that starts 2 words ahead of
    the keyword
    :param text: a text piece that may include a few sentences
    :param keyword: one word
    :return: a list of phrases
    """
    sentences = split_text(text)
    phrases = []
    keyword = word_process(keyword)
    for sentence in sentences:
        words = re.findall(r'\w+', sentence)
        for i, word in enumerate(words):
            if word_process(word) == word_process(keyword): # Both word and keyword have been processed, so we can compare them directly
                start = sentence.index(words[max(0,i-2)])
                end = sentence.index(word) + len(word)
                phrases.append(sentence[start:end])
    return phrases
