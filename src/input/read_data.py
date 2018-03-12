from src.support.support_functions import *
from spec.input_data_columns.data_inputs_spec import get_data_spec


input_data_path = 'input/'
columns = ['Store', 'Device', 'Source', 'Country', 'Date', 'Version', 'Rating', 'Original Reviews', 'Translated Reviews', 'Sentiment']


def read_all_data():
    """
    Function to read through all the datasets in the target folder
    """
    input_data_path = 'Input/'
    df = pd.DataFrame()
    file_paths = glob.glob(input_data_path + '*')
    for file_path in file_paths:
        df = read_input_file(file_path, df)
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
    data_spec = get_data_spec()

    def match_colnames(input_colname, data_source_name):
        colname_mapper = data_spec[data_source_name]['column_name_mapper']
        for key, colname in colname_mapper.items():
            if isinstance(colname, str) and len(colname) > 0:
                if not colname in input_colname:
                    return False
        return True

    for data_source_name in data_spec.keys():
        if match_colnames(df.columns, data_source_name):
            return data_source_name
    return 'Unknown'


def process_surveygizmo_df(df):
    """
    Function to Process the SurveyGizmo Dataframe
    """
    data_spec = get_data_spec()

    def extract_store_and_device(df, df_output):
        colname_mapper = data_spec['SurveyGizmo']['column_name_mapper']
        df_output['device_temp'] = df[colname_mapper['Device']]
        df_output['Store'] = ''
        df_output['Device'] = ''
        specified_devices = data_spec['SurveyGizmo']['device']
        device2store = data_spec['SurveyGizmo']['store']
        for i, row in df_output.iterrows():
            for device_name in specified_devices:
                if device_name in row['device_temp']:
                    row['Store'] = device2store[device_name]
                    row['Device'] = device_name
        df_output = df_output[df_output['Store'] != '']
        df_output = df_output.reset_index(drop=True)
        df_output = df_output.drop(['device_temp'], axis=1)
        return df_output

    def extract_version(df, df_output):
        """
        Function to extract the version information from the Corresponding Column in SurveyGizmo
        """
        df_output['version_temp'] = df[colname_mapper['Version']]
        df_output['Version'] = ''
        for i, row in df_output.iterrows():
            string = row['version_temp']  # Extract the string in the current row
            ios_locator = string.find("FxiOS/")  # Locate the ios-related term in string
            desktop_locator = string.find("Firefox/")  # Locate the desktop-related term in string
            if ios_locator > 0:  # Find the keyword
                version_code = string.split("FxiOS/", 1)[1].split(' ')[0]  # Example: 10.0b6373
                version = re.findall("^\d+\.\d+\.\d+|^\d+\.\d+", version_code)[
                    0]  # Extract the float number in the string with multiple dot
                digits = version.split('.')
                if len(digits) >= 2:  # 10.1 or 10.0.1
                    version = float(digits[0] + '.' + digits[1])
                    # Just capture the first two digits so that we can return as a number
                else:
                    version = int(version)
            elif desktop_locator > 0:
                version_code = string.split("Firefox/", 1)[1].split(' ')[0]  # Example: 57.0
                version = re.findall("^\d+\.\d+\.\d+|^\d+\.\d+", version_code)[
                    0]  # Extract the float number in the string with multiple dot
                digits = version.split('.')
                if len(digits) >= 2:  # 10.1 or 10.0.1
                    version = float(digits[0] + '.' + digits[1])
                    # Just capture the first two digits so that we can return as a number
                else:
                    version = int(version)
            else:
                version = 0
            row['Version'] = version
        df_output = df_output[df_output['Version'] != 0]
        df_output = df_output.drop(['version_temp'], axis=1)
        df_output = df_output.reset_index(drop=True)
        return df_output

    colname_mapper = data_spec['SurveyGizmo']['column_name_mapper']
    df_output = pd.DataFrame(index=range(len(df)))
    df_output = extract_store_and_device(df, df_output)
    df_output['Source'] = 'SurveyGizmo'
    df_output['Date'] = pd.to_datetime(df[colname_mapper['Date']]).dt.date
    df_output = extract_version(df, df_output)
    df_output['Original Reviews'] = df[colname_mapper['Original Reviews']].apply(
        lambda x: '{}{}'.format(x[0], x[1]), axis=1)
    df_output['Translated Reviews'] = ''
    df_output['Rating'] = ''
    df_output['Country'] = process_country(df[colname_mapper['Country']])
    return df_output


def process_appbot_df(df):
    """
    Function to Process the Appbot Dataframe
    """
    data_spec = get_data_spec()

    def extract_store_and_device(df, df_output):
        df_output['Store'] = df[colname_mapper['Store']]
        df_output['Device'] = 'Unknown'
        return df_output

    colname_mapper = data_spec['Appbot']['column_name_mapper']
    df_output = pd.DataFrame(index=range(len(df)))
    df_output = extract_store_and_device(df, df_output)
    df_output['Source'] = 'Appbot'
    df_output['Date'] = pd.to_datetime(df[colname_mapper['Date']]).dt.date
    df_output['Version'] = df[colname_mapper['Version']]
    df_output['Rating'] = df[colname_mapper['Rating']]
    df_output['Original Reviews'] = df[colname_mapper['Original Reviews']].apply(lambda x: '{}. {}'.format(x[0], x[1]),
                                                                             axis=1)
    df_output['Country'] = process_country(df[colname_mapper['Country']])
    return df_output


def process_country(Countries):
    """
    Process the country column in the two datasets
    :param Countries: a column of country value
    :return: cleaned country values
    """
    Countries.replace(to_replace=dict(USA='United States'), inplace=True)
    return Countries
