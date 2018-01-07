from support_functions import *

def summarize(df):
    summary_list = []
    # for i in range(len(df)):
    actions_list = df['Actions'].fillna('-')
    for i, row in df.iterrows():
        actions = actions_list[i]
        if len(str(actions)) <= 1:
            sentiment = row['Sentiment']
            summary = 'General ' + sentiment + ' Feedback'
        else:
            actions = actions.split(',')
            summary = actions[0]
        summary_list.append(summary)
    df['Issue Summary'] = summary_list
    return df

