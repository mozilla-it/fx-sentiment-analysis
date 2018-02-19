from support_functions import *


def print_outputs(target_folder_path, target_columns, n):
    outputs_path = target_folder_path + 'output/'
    df_categorization = pd.read_csv(outputs_path + 'categorization.csv')
    df_feedbacks = pd.read_csv(outputs_path + 'feedbacks.csv')
    df_tag = pd.read_csv(outputs_path + 'tag.csv')
    df_tag = df_tag.sort_values(by=['ID', 'Component'])
    df_join = df_feedbacks.merge(df_categorization, on='ID')
    df_join = df_join.merge(df_tag, on=['ID', 'Component'])


    for i, row in df_join.iterrows():
        if i < n:
            for target_col in target_columns:
                print(target_col)
                print('    ' + str(row[target_col]))
            print()


target_folder_path = 'Data/2017_12_28/'
target_columns = ['ID', 'Translated Reviews', 'Verb Phrases', 'Noun Phrases', 'Component', 'Tag']
n = 20
print_outputs(target_folder_path, target_columns, n)
