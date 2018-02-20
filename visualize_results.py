from support_functions import *


def list_to_array(l):
    output = ''

    for item in l:
        if isNaN(item):
            return 'None'
        output += item + ', '
    return output[:-2]


def print_outputs(target_folder_path, n):
    outputs_path = target_folder_path + 'output/'
    df_categorization = pd.read_csv(outputs_path + 'categorization.csv')
    df_feedbacks = pd.read_csv(outputs_path + 'feedbacks.csv')
    df_tag = pd.read_csv(outputs_path + 'tag.csv')
    df_tag = df_tag.sort_values(by=['ID', 'Component'])

    for i in range(min(len(df_feedbacks), n)):
        df_feedback_i = df_feedbacks[df_feedbacks['ID'] == i]
        print('Feedback No.' + str(i))
        print('Content:')
        print('    ' + list(df_feedback_i['Translated Reviews'])[0])
        print('Verb Phrases: ' + list_to_array(list(df_feedback_i['Verb Phrases'])))
        print('Noun Phrases: ' + list_to_array(list(df_feedback_i['Noun Phrases'])))
        print()
        df_categorization_i = df_categorization[df_categorization['ID'] == i]
        print('Find ' + str(len(df_categorization_i)) + ' component(s): ')
        for j in range(len(df_categorization_i)):
            component = list(df_categorization_i['Component'])[j]
            tags = list(df_tag[(df_tag['ID'] == i) & (df_tag['Component'] == component)]['Tag'])
            print('   Component #' + str(j+1) + ': ' + component + '; The corresponding Tag(s): ' + list_to_array(tags))

        print()
        print('==============')


target_folder_path = 'Data/2018_02_15/'
n = 4000
print_outputs(target_folder_path, n)
