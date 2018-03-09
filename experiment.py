from src.categorization import *
# sentence = "my email is ivan@gmail.com, and my phone number is 6472488190"


"""
df = pd.DataFrame([sentence], columns=['Translated Reviews'])

df = spam_filter(df)
print(list(df['Translated Reviews']))
print()

if len(df)> 0:
    df['ID'] = np.arange(len(df))  # Add ID Column
    df_categorization, df = categorize(df)
    print(df)
    print(df_categorization)
"""
target_folder_path = 'Data/2018_02_23/'
col_names = ['Store','Source','Date','Version','Rating','Original Reviews','Translated Reviews','Sentiment']
df = read_all_data(col_names, target_folder_path)
# df = spam_filter(df, colname='Original Reviews')
df = df[:200]
df = translate_reviews(df)  # Translate non-English reviews
print(len(df))
df_filtered = df[df['Translated Reviews'] == 'Error: no language detected!']
print(len(df_filtered))
df_filtered = df_filtered.reset_index(drop=True)
original_reviews = list(df_filtered['Original Reviews'])
for original_review in original_reviews:
    print(original_review)