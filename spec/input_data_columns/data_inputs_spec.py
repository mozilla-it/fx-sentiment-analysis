from spec.input_data_columns.appbot import get_appbot_column_names
from spec.input_data_columns.survey_gizmo import get_survey_gizmo_columns


def get_data_spec():
    return {
        'SurveyGizmo': {
            'column_name_mapper': get_survey_gizmo_columns(),  # Call the column name mapper
            'device': ['iPhone', 'iPad', 'Windows', 'Macintosh', 'Linux'],  # Will only look at the specified devices
            'store': {
                'iPhone': 'iOS',
                'iPad': 'iOS',
                'Windows': 'Desktop',
                'Macintosh': 'Desktop',
                'Linux': 'Desktop'
            }
        },
        'Appbot': {
            'column_name_mapper': get_appbot_column_names(),  # Call the column name mapper
            'device': [],
            'store': 'iOS'  # Label
        }
    }