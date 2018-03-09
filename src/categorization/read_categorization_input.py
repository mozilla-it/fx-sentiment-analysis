# First, you're going to need to import wordnet:
from src.support.support_functions import *


# global cate_file_path
cate_file_path = 'Spec/categorization_keywords.csv'

class CategorizationDict:
    def __init__(self, features, components, keywords, components2tiers, components2features, keywords2components):
        self.features = features  # Dictionary that maps components to their features
        self.components = components
        self.keywords = keywords  # List of Keywords
        self.components2tiers = components2tiers  # Dictionary that maps components to their tiers
        self.components2features = components2features # Dictionary that maps components to their features
        self.keywords2components = keywords2components # Dictionary that maps keywords to components


def read_categorization_file(cate_file_path):
    return pd.read_csv(cate_file_path)


def update_keyword_to_component(keyword, component, keywords, keywords2components):
    if len(keyword) > 0:  # prevent empty keyword
        if keyword in keywords2components.keys():
            if isinstance(keywords2components[keyword], list):
                keywords2components[keyword].append(component)
            else:
                first_component = keywords2components[keyword]
                keywords2components[keyword] = [first_component, component]
        else:
            keywords.append(keyword)
            keywords2components[keyword] = component
    return keywords, keywords2components


def get_categorization_input():
    """
    Function to read content from the categorization file
    :return: cate: an instance of categorization that stores all the pre-defined content
    """
    df = read_categorization_file(cate_file_path)

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
        keywords_of_component = split_input_words(row['Keywords']) if not(isNaN(row['Keywords'])) else []
        keyword_list_from_component = get_keywords_for_component(component, keywords_of_component)
        for keyword in keyword_list_from_component:
            keywords, keywords2components = update_keyword_to_component(keyword,
                                                                        component,
                                                                        keywords,
                                                                        keywords2components)

        keywords = list(set(keywords))
    cateDict = CategorizationDict(features, components, keywords, components2tiers, components2features, keywords2components)
    return cateDict


def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            syn_word = l.name()
            syn_word = syn_word.replace("_", " ")
            synonyms.append(syn_word)
    synonyms = list(set(synonyms))
    synonyms = select_most_similar_words(word, synonyms)
    return synonyms


def select_most_similar_words(keyword, synonyms, thresh=0.75, n=3):
    if len(synonyms) <= n:
        return synonyms

    if keyword in synonyms:
        synonyms.remove(keyword)

    similarity_list = []
    for i, word in enumerate(synonyms):
        sim = compute_similarity_between_words(keyword, word)
        similarity_list.append(sim)
    if len(synonyms) <= n:
        return synonyms
    synonyms_selected, sim_selected = select_from_list_a_based_on_list_b(synonyms, similarity_list,
                                                                         min_thresh=thresh, k=n)
    return synonyms_selected


def get_keywords_for_component(component, user_defined_keywords):
    """
    Process the keywords for the given component
    :return: a list of keywords for the given component
    """
    output_keywords = []
    input_keywords = user_defined_keywords  # initialize with the user defined keywords
    input_keywords += component.split('/')  # split the component if there are multiple terms involved
    for input_keyword in input_keywords:
        output_keywords.append(input_keyword)
        word_list_split_by_space = input_keyword.split(' ')
        for word in extract_words_from_word_list_split_by_space(word_list_split_by_space):
            output_keywords.append(word)
            output_keywords += get_synonyms(word)
    output_keywords = list(set(output_keywords))
    return output_keywords


def extract_words_from_word_list_split_by_space(word_list):
    """
    For the given phrases, we cannot use each individual word as the keyword; thus, we need to define rules of breaking
    phrases
    """
    words = generate_pairs_of_words(word_list)
    return words


def generate_pairs_of_words(word_list):
    """
    Given a list of words, generate all potential combination of words in pairs
    :param word_list:
    :return: a list of words in pairs
    """
    def pair_words(word_list, i, j, connector):
        return word_list[i] + connector + word_list[j]
    pairs = []
    n = len(word_list)
    for i in range(n-1):
        for j in range(i+1, n):
            pairs.append(pair_words(word_list, i, j, ' '))
            pairs.append(pair_words(word_list, j, i, ' '))
            pairs.append(pair_words(word_list, i, j, '-'))
            pairs.append(pair_words(word_list, j, i, '-'))
            pairs.append(pair_words(word_list, i, j, '_'))
            pairs.append(pair_words(word_list, j, i, '_'))
            pairs.append(pair_words(word_list, i, j, ''))
            pairs.append(pair_words(word_list, j, i, ''))
    outputs = list(set(pairs))  # remove duplicates
    return outputs

"""
cateDict = get_categorization_input()
keywords_list = []
for component in cateDict.components:
    keywords = []
    for keyword in cateDict.keywords:
        if cateDict.keywords2components[keyword] == component:
            keywords.append(keyword)
    keywords_list.append(keywords)
df = pd.DataFrame({
    'Components': cateDict.components,
    'Keywords': keywords_list
})
df.to_csv('Auto_Generated_keywords.csv')
"""