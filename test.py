from Categorization import *
from Cluster_Summarize import *


def spam_filter(df, colname = 'Translated Reviews'):
    df_copy = df.copy()
    spams = np.zeros(len(df))
    for i, row in df.iterrows():
        feedback = row[colname]
        too_long = (len(feedback) > 1000) & (len(re.findall(r'[Ff]irefox|browser', feedback)) < 2)
        emails_match = len(re.findall(r'[\w\.-]+@[\w\.-]+', feedback))
        too_many_digits = len(re.findall(r'\d+', feedback)) > 30
        if (too_long + emails_match + too_many_digits > 0):
            spams[i] =1
    df_copy['Spam'] = spams
    df = df[df_copy['Spam'] == 0]
    df.to_csv('temp2.csv', index=False)
    df = pd.read_csv('temp2.csv')
    return df

data_path = 'Data/2017_11_30/output_py.xlsx'
target_folder_path = 'Data/2017_11_30/'
df = read_exist_output(data_path)
# df = pd.read_csv('temp.csv')
df = spam_filter(df)
df.to_csv('temp2.csv', index=False)
df = pd.read_csv('temp2.csv')
df = index_df(df)
df.to_csv('temp2.csv', index=False)
df_categorization = categorize(df)
df_categorization.to_csv('temp_cate2.csv', index=False)
df_categorization = cluster_and_summarize(df, df_categorization)
df_categorization.to_csv('temp_cate2.csv', index=False)

# df = pd.read_csv('temp.csv')

"""
df_features_counts = freq_count(df, 'Features')
df_comp_counts = freq_count(df, 'Components')
df_actions_counts = freq_count(df, 'Actions')

# Save into an output file in the target folder

output_path = target_folder_path + 'output_py.xlsx'
writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
# df.to_csv(output_path,encoding='utf-8')
df.to_excel(writer, sheet_name='Full', index=False)
df_features_counts.to_excel(writer, sheet_name='Feature Count', index=False)
df_comp_counts.to_excel(writer, sheet_name='Component Count', index=False)
df_actions_counts.to_excel(writer, sheet_name='Action Count', index=False)

writer.save()
print('Output has been saved to: ' + target_folder_path)
"""