import json

column_name_mapper = {}

column_name_mapper['SurveyGizmo'] = {
    'Store': '',
    'Date': 'Date Submitted',
    'Version': 'Extended User Agent',
    'Rating': '',
    'Original Review': [
        'To help us understand your input, we need more information. Please describe what you like. The content of your feedback will be public, so please be sure not to include personal information such as email address, passwords or phone number.',
        'To help us understand your input, we need more information. Please describe your problem below and be as specific as you can. The content of your feedback will be public, so please be sure not to include personal information such as email address, passwords or phone number.'
    ],
    'Country': 'Country',
}

column_name_mapper['Appbot'] = {
    'Store': 'Store',
    'Date': 'Date',
    'Version': 'Version',
    'Rating': 'Rating',
    'Original Reviews': ['Subject', 'Body'],
    'Country': 'Country',
}

with open('spec/input_data_columns.txt', 'w') as outfile:
    json.dump(column_name_mapper, outfile)

