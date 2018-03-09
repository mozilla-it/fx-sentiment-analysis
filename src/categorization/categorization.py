from src.support.support_functions import *
from src.categorization.read_categorization_input import get_categorization_input


def list_to_string(input_list):
    inputs = np.hstack(input_list)
    # inputs = list(set(inputs))
    output_string = ''
    for i, item in enumerate(inputs):
        if i > 0:
            output_string += ' || '
        output_string += item
    return output_string


def clean_components(components):
    """
    This function defines all the pre_processing procedures to components identified
    :param components: a list of components found in a piece of feedback
    :return: a list of components after pre_processing
    """
    if len(components) > 1 and 'Firefox Browser' in components:
        components.remove('Firefox Browser')
    return components


def categorize(df):
    cateDict = get_categorization_input() # Read the content from the categorization file

    df['Verb Phrases'] = extract_phrases(df['Translated Reviews'], 'VP')
    df['Noun Phrases'] = extract_phrases(df['Translated Reviews'], 'NP')

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
            keywords_found = ''
            if (len(keywords_found_VP_list[i]) > 0 or len(keywords_found_NP_list[i]) > 0):
                keywords_found += keywords_found_VP_list[i]
                if len(keywords_found_NP_list[i]) > 0 :
                    if len(keywords_found) > 0:
                        keywords_found += ', '
                    keywords_found += keywords_found_NP_list[i]
            else:
                keywords_found += keywords_found_text_list[i]

            for keyword in keywords_found.split(', '):
                if len(keywords_found_VP_list[i]) > 0:
                    actions_found = [VP for VP in str(row['Verb Phrases']).split(', ') if
                                    word_process(keyword) in phrase_process(VP)]
                else:
                    actions_found = extract_phrases_with_keywords(str(row['Translated Reviews']), keyword)

                if not isinstance(cateDict.keywords2components[keyword], list):
                    keyword_component_list = [cateDict.keywords2components[keyword]]
                else:
                    keyword_component_list = cateDict.keywords2components[keyword]
                for component_found in keyword_component_list:
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
        components_found = clean_components(components_found)
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
            'Actions': actions_found_list
        }
    )
    df_categorization = df_categorization.replace(np.nan, '', regex=True)
    return df_categorization, df


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
            if word_process(word) == word_process(keyword):  # Both word and keyword have been processed, so we can compare them directly
                start = sentence.index(words[max(0,i-2)])
                end = sentence.index(word) + len(word)
                phrases.append(sentence[start:end])
    return phrases