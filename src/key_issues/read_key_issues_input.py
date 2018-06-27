from src.categorization.read_categorization_input import *


# global cate_file_path
key_issue_file_path = 'spec/key_issues_keywords.csv'

class KeyIssueDict:
    def __init__(self, key_issues, keywords, keyword2issues):
        self.key_issues = key_issues
        self.keywords = keywords
        self.keyword2issues = keyword2issues


def read_key_issue_file(key_issue_file_path):
    return pd.read_csv(key_issue_file_path)


def process_keywords_for_bug_issue(keywords):
    """
    A function dedicated to enrich the keywords for bug issue
    :param keywords: a list of keywords given by users (Verbs)
    :return: an enriched list of keywords
    """
    auxiliary_verb_lib = "doesn’t, doesnot, does not, didn't, did not, don't, donot, do not, can't, cannot, can not, " \
                         "couldn't, couldnot, could not, will not, won't, wouldn't, would not, wouldnot, isn't, is not," \
                         "aren't, wasn't, weren't"
    auxiliary_verb_list = split_input_words(auxiliary_verb_lib)
    output_keywords_list = ['bug']
    for verb in auxiliary_verb_list:
        for keyword in keywords:
            output_keywords_list.append(verb + ' ' + keyword)
    return output_keywords_list


def get_key_issue_input(store_name):
    df_master = read_key_issue_file(key_issue_file_path)
    df = df_master[df_master['Store'] == store_name]
    key_issues = list(df['Issue'])

    keywords = []
    keyword2issues = {}
    for i, row in df.iterrows():
        issue = row['Issue']
        keywords_of_issue = split_input_words(row['Keywords']) if not (isNaN(row['Keywords'])) else []
        if issue == 'Bug':
            keyword_list_of_issue = process_keywords_for_bug_issue(keywords_of_issue)
        else:
            keyword_list_of_issue = get_keywords_for_component(issue, keywords_of_issue)
        for keyword in keyword_list_of_issue:
            keywords, keyword2issues = update_keyword_to_component(keyword,
                                                                        issue,
                                                                        keywords,
                                                                   keyword2issues)
        keywords = list(set(keywords))
    return KeyIssueDict(key_issues, keywords, keyword2issues)
