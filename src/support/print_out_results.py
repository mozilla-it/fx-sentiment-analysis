from src.data_ouptut.sqlite import extract_contents_from_db


def list_to_array(l):
    output = ''

    for item in l:
        try:
            output += item + ', '
        except TypeError:
            return 'None'
    return output[:-2]


def print_outputs():
    df_feedbacks, df_categorization, df_key_issue = extract_contents_from_db()
    print_contents(df_categorization, df_feedbacks, df_key_issue)


def print_contents(df_categorization, df_feedbacks, df_key_issue, i_max=50):
    df_key_issue = df_key_issue.sort_values(by=['ID'])
    min_id = min(df_feedbacks['ID'])
    max_id = max(df_feedbacks['ID'])

    for i in range(min_id, max_id+1):
        if sum(df_feedbacks['ID'] == i) > 0:
            df_feedback_i = df_feedbacks[df_feedbacks['ID'] == i]
            print('Feedback No.' + str(i))
            print('Content:')
            print('    ' + str(list(df_feedback_i['Translated Reviews'])[0]))
            print('Verb Phrases: ' + list_to_array(list(df_feedback_i['Verb Phrases'])))
            print('Noun Phrases: ' + list_to_array(list(df_feedback_i['Noun Phrases'])))
            print()
            df_categorization_i = df_categorization[df_categorization['ID'] == i]
            print('Find ' + str(len(df_categorization_i)) + ' component(s): ')
            for j in range(len(df_categorization_i)):
                component = list(df_categorization_i['Component'])[j]

                print('   Component #' + str(j + 1) + ': ' + component + '; ')
            key_issues = list(df_key_issue[df_key_issue['ID'] == i]['Issue'])
            print('Issues:')
            print(key_issues)

            print()
            print('==============')


if __name__ == '__main__':
    print_outputs()