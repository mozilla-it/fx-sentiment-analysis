from support_functions import *


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


def cluster_and_summarize(df, df_categorization):
    df_join = df.merge(df_categorization, on='ID')
    df_categorization['Issue Summary'] = ''
    components = df_join.Component.unique()
    for component in components:
        df_selected = df_join[df_join['Component'] == component]
        feedbacks = df_selected['Translated Reviews']
        similarity_matrix = measure_sim_tfidf(feedbacks, viz=False)

        thresh = 0.3
        indice_list = []
        for index, x in np.ndenumerate(np.triu(similarity_matrix, k=1)):
            if x >= thresh:
                indice_list.append(list(index))
        clusters = merge_overlapped_lists(indice_list)
        if len(clusters) > 0:
            remaining = list(set(np.arange(len(df_selected))) - set(np.hstack(clusters)))
            joint_clusters = clusters + remaining
        else:  # No cluster is found
            joint_clusters = np.arange(len(df_selected))

        for cluster in joint_clusters:
            if not isinstance(cluster, list):
                cluster = [cluster]
            actions = []
            for i in cluster:
                # print(df_selected['Action'].iloc[i])
                if isinstance(df_selected['Action'].iloc[i], list):
                    actions += df_selected['Action'].iloc[i]
            actions_list = list(set(actions))

            issue_summary = summarize(actions_list)

            for i in cluster:
                ID = df_selected['ID'].iloc[i]
                df_categorization.at[(df_categorization['ID'] == ID) & (df_categorization['Component'] == component), 'Issue Summary'] = issue_summary
    return df_categorization




