# First, you're going to need to import wordnet:
from support_functions import *


# global cate_file_path
cate_file_path = 'Data/Categorization.csv'

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


def get_categorization_input():
    """
    Function to read content from the categorization file
    :return: cate: an instance of categorization that stores all the pre-defined content
    """
    df = read_categorization_file(cate_file_path)

    # Ensure there are no duplicate in components
    assert sum(df['Component'].duplicated()) == 0, \
        "There are duplicated entry in the column Components in the file: %r" % cate_file_path

    components = df['Component'].unique()  # Read components
    auto_generated_keywords_dict = get_keywords_for_components(components)
    features = df['Feature'].unique()

    keywords = []
    components2tiers = {}
    components2features = {}
    keywords2components = {}
    for i, row in df.iterrows():
        component = row['Component']
        components2tiers[component] = row['Tier']
        components2features[component] = row['Feature']
        # User-defined keywords
        for keyword in row['Keywords'].split(', '):
            keywords.append(keyword)
            keywords2components[keyword] = component

        # System-generated keywords: based on synonyms from WordNet
        for keyword in auto_generated_keywords_dict[component]:
            keywords.append(keyword)
            keywords2components[keyword] = component

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


def get_keywords_for_component(component):
    keywords = [component]  # initialize with the input word itself
    component_word_list_split_1 = component.split('/')
    for component_word in component_word_list_split_1:
        keywords.append(component_word)
        component_word_list_split_2 = component_word.split(' ')
        for word in component_word_list_split_2:
            keywords.append(word)
            keywords += get_synonyms(word)
    keywords = list(set(keywords))
    return keywords


def get_keywords_for_components(components):
    keywords_list = []
    for component in components:
        keywords = get_keywords_for_component(component)
        for keyword in keywords:
            for i in range(len(keywords_list)):
                if keyword in keywords_list[i]:
                    keywords.remove(keyword)
                    keywords_list[i].remove(keyword)
        keywords_list.append(keywords)
    keywords_dict = {}
    for i, component in enumerate(components):
        keywords_dict[component] = keywords_list[i]
    return keywords_dict


df = read_categorization_file(cate_file_path)
components = df['Component'].unique()  # Read components
auto_generated_keywords_dict = get_keywords_for_components(components)