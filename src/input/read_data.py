from src.support_functions import *
from spec.input_data_columns.appbot import get_appbot_column_names
from spec.input_data_columns.survey_gizmo import get_survey_gizmo_columns


input_data_path = 'input/'
columns = ['Store', 'Source', 'Country', 'Date', 'Version', 'Rating', 'Original Reviews', 'Translated Reviews', 'Sentiment']
global data_specs
data_specs = {
    'SurveyGizmo':{
        'column_name_mapper': get_survey_gizmo_columns(),
        'dates_covered': []
    },
    'Appbot':{
        'column_name_mapper': get_appbot_column_names(),
        'dates_covered': []
    }
}


def read_all_data(date_threshold):
    """
    Function to read through all the datasets in the target folder
    """
    df = pd.DataFrame()
    # Read in all the dataset
    file_paths = glob.glob(input_data_path + '*')
    for file_path in file_paths:
        df = read_input_file(file_path, df)
    df = filter_by_date(df, date_threshold)
    return df


def read_input_file(file_path, df):
    """
    Function to read in the input file
    :param file_path:
    :param df: dataframe of feedbacks
    :return: updated df
    """
    file_format = file_path.split('.')[-1]
    if file_format == 'xlsx':
        xl = pd.ExcelFile(file_path)
        df_input = xl.parse(xl.sheet_names[0]).fillna('')
    elif file_format == 'csv':
        df_input = pd.read_csv(file_path).fillna('')
    else:
        print('The input file format ' + file_format + ' is not supported!')
        return None

    data_source = identify_data_source(df_input)
    if data_source == 'Unknown':
        print('Cannot identify the source of the input data ' + file_path + ', please check the column names@')
    elif data_source =='SurveyGizmo':
        df_processed = process_surveygizmo_df(df_input)
    elif data_source == 'Appbot':
        df_processed = process_appbot_df(df_input)
    return pd.concat([df, df_processed])  # Merged the dataframes


def identify_data_source(df):
    """
    Categorize the input dataframe into Appbot/SurveyGizmo
    :param df: input dataframe
    :return: name of the data source in string
    """
    def match_colnames(input_colname, data_source_name):
        colname_mapper = data_specs[data_source_name]['column_name_mapper']
        for key, colname in colname_mapper.items():
            if isinstance(colname, str) and len(colname) > 0:
                if not colname in input_colname:
                    return False
        return True

    for data_source_name in data_specs.keys():
        if match_colnames(df.columns, data_source_name):
            return data_source_name
    return 'Unknown'


def process_surveygizmo_df(df):
    """
    Function to Process the SurveyGizmo Dataframe
    """

    def extract_version_SG(SG_Col):
        """
        Function to extract the version information from the Corresponding Column in SurveyGizmo
        """
        version_list = []
        for i in range(len(SG_Col)):
            string = SG_Col[i]  # Extract the string in the current row
            locator = string.find("FxiOS/")  # Locate the target term in each string
            if locator > 0:  # Find the keyword
                version_code = string.split("FxiOS/", 1)[1].split(' ')[0]  # Example: 10.0b6373
                version = re.findall("^\d+\.\d+\.\d+|^\d+\.\d+", version_code)[
                    0]  # Extract the float number in the string with multiple dot
                digits = version.split('.')
                if len(digits) >= 2:  # 10.1 or 10.0.1
                    version = float(digits[0] + '.' + digits[
                        1])  # Temporary use - just capture the first two digits so that we can return as a number
                else:
                    version = int(version)
            else:
                version = 0
            version_list.append(version)
            # print('Origin: ' + string + ', Version: ' + str(version))
        return version_list


    colname_mapper = data_specs['SurveyGizmo']['column_name_mapper']
    df_output = pd.DataFrame(index=range(len(df)))
    df_output['Store'] = 'iOS'
    df_output['Source'] = 'SurveyGizmo'
    df_output['Date'] = pd.to_datetime(df[colname_mapper['Date']]).dt.date
    df_output['Version'] = extract_version_SG(df[colname_mapper['Version']])
    df_output['Original Reviews'] = df[colname_mapper['Original Reviews']].apply(
        lambda x: '{}{}'.format(x[0], x[1]), axis=1)
    df_output['Translated Reviews'] = ''
    df_output['Country'] = process_country(df[colname_mapper['Country']])
    df_output = update_dates_processed(df_output, 'SurveyGizmo')
    return df_output


def process_appbot_df(df):
    """
    Function to Process the Appbot Dataframe
    """
    colname_mapper = data_specs['Appbot']['column_name_mapper']
    df_output = pd.DataFrame(index=range(len(df)))
    df_output['Store'] = 'iOS'
    df_output['Source'] = 'Appbot'
    df_output['Date'] = pd.to_datetime(df[colname_mapper['Date']]).dt.date
    df_output['Version'] = df[colname_mapper['Version']]
    df_output['Rating'] = df[colname_mapper['Rating']]
    df_output['Original Reviews'] = df[colname_mapper['Original Reviews']].apply(lambda x: '{}. {}'.format(x[0], x[1]),
                                                                             axis=1)
    df_output['Country'] = df[colname_mapper['Country']]
    df_output = update_dates_processed(df_output, 'Appbot')
    return df_output


def process_country(Countries):
    """
    Process the country column in the two datasets
    :param Countries: a column of country value
    :return: cleaned country values
    """
    Countries.replace(to_replace=dict(USA='United States'), inplace=True)
    return Countries


def update_dates_processed(df, source_name, colname='Date'):
    """
    Remove dates that have been processed, and update the list of processed datas
    :param df:
    :param source_name:
    :param colname:
    :return:
    """
    dates_in_process = df[colname].unique()
    for date in dates_in_process:
        if date in data_specs[source_name]['dates_covered']:  # This date has been processed
            df[df[colname] != date]  # Remove the correponding content
        else:
            data_specs[source_name]['dates_covered'].append(date)  # Mark this date as been processed
    df = df.reset_index(drop=True)  # Reset the index as we have removed contents
    return df


def filter_by_date(df, date_threshold):
    """
    The function remove rows whose date is before the given date threshold
    """
    date = datetime.strptime(date_threshold, '%Y-%m-%d').date()  # Convert the given threshold (in string) to date
    df_filtered = df[df['Date'] >= date]
    df = df.reset_index(drop=True)  # Reset the index as we have removed contents
    return df_filtered