from support_functions import *

global cate_file_path
cate_file_path = 'Data/Categorization.csv'


def list_to_string(input_list):
    inputs = np.hstack(input_list)
    # inputs = list(set(inputs))
    output_string = ''
    for i, item in enumerate(inputs):
        if i > 0:
            output_string += ' || '
        output_string += item
    return output_string


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
    row_id_list = []
    features_found_list = []
    components_found_list = []
    actions_found_list = []
    for i, row in df.iterrows():
        components_found = []
        components_dict = {}
        if len(keywords_found_text_list[i]) == 0:  # No keywords can be identified in the entire text
            component_found = 'Others'
            components_found = [component_found]
            components_dict[component_found] = {
                'Feature': 'Other',
                'Action': ['']
            }
        else:
            if len(keywords_found_VP_list[i])  > 0:
                keywords_found = keywords_found_VP_list[i]
            elif len(keywords_found_NP_list[i])  > 0:
                keywords_found = keywords_found_NP_list[i]
            else:
                keywords_found = keywords_found_text_list[i]

            for keyword in keywords_found.split(', '):
                if len(keywords_found_VP_list[i]) > 0:
                    actions_found = [VP for VP in str(row['Verb Phrases']).split(', ') if
                                    word_process(keyword) in phrase_process(VP)]
                else:
                    actions_found = extract_phrases_with_keywords(str(row['Translated Reviews']), keyword)

                component_found = cateDict.keywords2components[keyword]
                if component_found in components_found:
                    for action in actions_found:
                        if action not in components_dict[component_found]['Action']:
                            components_dict[component_found]['Action'].append(action)
                else:
                    components_found.append(component_found)
                    components_dict[component_found] = {
                        'Feature': cateDict.components2features[component_found],
                        'Action': actions_found
                    }

        for component in components_found:
            row_id_list.append(row['ID'])
            components_found_list.append(component)
            features_found_list.append(components_dict[component]['Feature'])
            actions_found_list.append(components_dict[component]['Action'])


    df_categorization = pd.DataFrame(
        {
            'ID': row_id_list,
            'Feature': features_found_list,
            'Component': components_found_list,
            'Action': actions_found_list
        }
    )
    df_categorization = df_categorization.replace(np.nan, '', regex=True)
    return df_categorization


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
