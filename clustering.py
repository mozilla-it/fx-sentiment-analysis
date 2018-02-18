from support_functions import *
from read_categorization_input import get_categorization_input

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


def summarize(text_list):
    if len(text_list) > 0:  #
        try:
            for text in text_list:
                if len(text) > 80:  # Should not be very long
                    text_list.remove(text)
            tfidf_score = np.mean(vectorize_tfidf(text_list).toarray(), axis=1)
            summary = text_list[np.argmax(tfidf_score)]
        except ValueError:
            summary = 'Unknown'
    else:
        summary = 'Unknown'
    return summary


def create_cluster(df, thresh = 0.3):
    texts = df['Translated Reviews']
    similarity_matrix = measure_sim_tfidf(texts, viz=False)
    indice_list = []
    for index, x in np.ndenumerate(np.triu(similarity_matrix, k=1)):
        if x >= thresh:
            indice_list.append(list(index))
    clusters = merge_overlapped_lists(indice_list)
    if len(clusters) > 0:
        remaining = list(set(np.arange(len(df))) - set(np.hstack(clusters)))
        joint_clusters = clusters + remaining
    else:  # No cluster is found
        joint_clusters = np.arange(len(df))
    return joint_clusters


def extract_keywords(texts, top_k=5):
    categorization_Dict = get_categorization_input()
    pre_defined_keywords_list = categorization_Dict.keywords

    if len(texts) < 5:
        top_k = 1
    keywords, counts = compute_keywords_freq(texts, k=top_k,
                                             additional_stop_words=pre_defined_keywords_list,
                                             get_counts=True)
    if len(keywords) > 0:
        keywords_selected = select_keywords(keywords.tolist(), counts.tolist(), texts)
        keywords_recovered = recover_keywords(keywords_selected, texts)
        return keywords_recovered
    else:
        return []


def recover_keywords(keywords, texts):
    keywords_recovered = []
    for keyword in keywords:
        matched_word = match_words(keyword, texts)
        if not matched_word == '':
            keywords_recovered.append(matched_word)
    return keywords_recovered


def match_words(keyword, texts):
    for text in texts:
        for word in re.findall(r'\w+', text):
            if keyword == word_process(word):
                return word
    return ''


def select_keywords(keywords, counts, texts):
    """
    The function manage the selection of the top keywords from texts
    :param keywords: a list of keywords extracted from texts
    :param counts: corresponding counts
    :param texts: original texts
    :return: selected keywords
    """
    lower_threshold = len(texts) * 0.8
    n = np.sum(np.array(counts) > lower_threshold)
    selection = max(1, n)
    return keywords[:selection]


def update_df(df, ID, component, keywords):
    existing_keywords = df.loc[(df['ID'] == ID) & (df['Component'] == component)]['Tags'].tolist()
    if not existing_keywords == ['']:
        keywords += existing_keywords
    if len(keywords) > 1:
        keywords = list(set(keywords))
        keywords_array = keywords[0]
        for i in range(1, len(keywords)):
            keywords_array += ', ' + keywords[i]
    else:
        keywords_array = keywords[0]
    df.at[(df['ID'] == ID) & (df['Component'] == component), 'Tags'] = keywords_array


def prepare_text(df, cluster):
    VP = [df['Verb Phrases'].iloc[i] for i in cluster]
    NP = [df['Noun Phrases'].iloc[i] for i in cluster]
    Actions = [df['Actions'].iloc[i] for i in cluster]
    texts = []
    for i in range(len(cluster)):
        text = ''
        text = text + VP[i] + ', ' if isinstance(VP[i], str) else text
        text = text + NP[i] + ', ' if isinstance(NP[i], str) else text
        text = text + Actions[i] + ', ' if isinstance(Actions[i], str) else text
        if len(text) > 0:
            texts.append(text)
    # if len(texts) == 0:
        # texts = [df['Translated Reviews'].iloc[i] for i in cluster]
    return texts


def cluster_and_summarize(df_feedbacks, df_categorization):
    df_join = df_feedbacks.merge(df_categorization, on='ID')
    df_categorization['Tags'] = ""
    components = df_join.Component.unique()
    for component in components:
        df_selected = df_join[df_join['Component'] == component]
        clusters = create_cluster(df_selected)
        for cluster in clusters:
            cluster = [cluster] if not (isinstance(cluster, list)) else cluster
            texts = prepare_text(df_selected, cluster)
            keywords = extract_keywords(texts)
            if len(keywords) > 0:
                for i in cluster:
                    ID = df_selected['ID'].iloc[i]
                    update_df(df_categorization, ID, component, keywords)
    return df_categorization
