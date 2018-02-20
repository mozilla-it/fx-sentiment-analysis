from support_functions import *
from read_categorization_input import get_categorization_input


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


def create_cluster(df, thresh=0.3):
    texts = df['Translated Reviews']
    texts_processed = process_texts_for_clustering(texts)
    similarity_matrix = measure_sim_tfidf(texts_processed, viz=False)
    return cluster_based_on_similarity(similarity_matrix, thresh)


def extract_keywords(texts, top_k=4):
    categorization_Dict = get_categorization_input()
    pre_defined_keywords_list = categorization_Dict.keywords

    if len(texts) < 5:
        top_k = 1
    keywords, counts = compute_keywords_freq(texts, additional_stop_words=pre_defined_keywords_list)
    if len(keywords) > 0:
        keywords_selected = select_keywords(keywords.tolist(), counts.tolist(), top_k)
        keywords_recovered = recover_words_from_texts(keywords_selected, texts)
        return keywords_recovered
    else:
        return []


def select_keywords(keywords, counts, top_k):
    """
    The function manage the selection of the top keywords from texts
    :param keywords: a list of keywords extracted from texts
    :param counts: corresponding counts
    :param texts: original texts
    :return: selected keywords
    """
    k = min(len(keywords), top_k)
    top_k_ranks = np.argsort(np.array(counts))[-k:]
    keywords_selected = [keywords[i] for i in top_k_ranks]
    return keywords_selected


def update_df_tag(df, ID, component, tags):
    for tag in tags:
        df.loc[len(df)] = [ID, component, tag]
    return df


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


def recover_words_from_texts(keywords, texts):
    keywords_recoverd = []
    for keyword in keywords:
        keywords_recoverd.append(recover_word_from_texts(keyword, texts))
    return keywords_recoverd


def recover_word_from_texts(keyword, texts):
    word_recovered = keyword
    words_candidates = []
    for text in texts:
        for word in re.findall(r'\w+', text):
            if keyword == word_process(word):
                words_candidates.append(word)
    counter = Counter(words_candidates)
    if len(counter) > 0:
        words, counts = get_counter_contents(counter, sorted=True)
        top_rank = np.argsort(np.array(counts))[-1:][0]
        word_recovered = words[top_rank]
    return word_recovered


def add_head_word_to_texts(head_word, words, processed_text_list):
    """
    Function to add the head_word to all the texts that contain a word in the given words
    :param head_word: a head word to be inserted into texts (processed)
    :param words: a list of words, including the head word (processed)
    :param processed_text_list: a list of processed texts (each text is a string)
    :return: processed_text_list after insertion
    """
    words.remove(head_word)
    n_texts = len(processed_text_list)
    for word in words:
        for i in range(n_texts):
            text_in_list = re.findall(r'\w+', processed_text_list[i])
            if word in text_in_list:
                processed_text_list[i] += ' ' + head_word
    return processed_text_list


def process_texts_for_clustering(texts):
    """
    Process texts with stemmer, remove stop words, and most importantly,
    find semantically similar words and manipulate their occurrence
    :param texts: a list of texts
    :return: a list of processed texts in strings
    """
    # Process texts
    texts_processed = process_texts(texts)

    # Find cluster of semantically similar words
    keywords, keywords_counts = select_keywords_on_freq(texts_processed,
                                                        min_thresh=4,
                                                        k=9999,
                                                        get_counts=True)
    keywords_recoverd = recover_words_from_texts(keywords, texts)
    similarity_matrix = compute_similarity_words(keywords_recoverd)
    clusters = cluster_based_on_similarity(similarity_matrix, thresh=0.6)

    for cluster in clusters:
        if len(cluster) > 1:
            words = [keywords[i] for i in cluster]
            words_counts = [keywords_counts[i] for i in cluster]
            head_word = words[np.argmax(words_counts)]  # Word with the most counts
            texts_processed = add_head_word_to_texts(head_word, words, texts_processed)
    return texts_processed


def cluster_and_summarize(df_feedbacks, df_categorization):
    df_join = df_feedbacks.merge(df_categorization, on='ID')
    df_tag = pd.DataFrame(columns=['ID', 'Component', 'Tag'])
    components = df_join.Component.unique()
    for component in components:
        df_selected = df_join[df_join['Component'] == component]
        clusters = create_cluster(df_selected)
        for cluster in clusters:
            texts = prepare_text(df_selected, cluster)
            keywords = extract_keywords(texts)
            if len(keywords) > 0:
                for i in cluster:
                    ID = df_selected['ID'].iloc[i]
                    update_df_tag(df_tag, ID, component, keywords)
    return df_categorization, df_tag