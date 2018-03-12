# User Feedback Analytics Tool
This tool is developed by the UofT Capstone team to help clients in Mozilla analyze their user feedbacks more efficiently and effectively.

## Required Environment and Packages
- Python 3.6
- nltk 3.2.5 or upper
- numpy 1.14.0 or upper
- pandas 0.22.0
- scikit-learn 0.19.1
- scipy 1.0.0
- sqlite 3.22.0

### Google Cloud API
The tool uses Google Translation API for translation and Google Natural Language API for sentiment analysis. Therefore, it requires a setup of Google Cloud in the local environment. 

First, you need to register for the two APIs in [Google Cloud Platform](https://console.cloud.google.com/).

Second, you need to install Google Cloud SDK as described in the precedure on the [Website](https://cloud.google.com/sdk/downloads)

Third, you need to set up the Authentication in the local machine, as described [here](https://cloud.google.com/docs/authentication/getting-started). It is encouraged to set up the "GOOGLE_APPLICATION_CREDENTIALS" as an environmental variable.

## To run the job
### Input Data
For the input data, I included a few user feedback files from both Appbot and SurveyGizmo in the folder [Input](Input/). 

### Data Pipeline
The data pipeline has been build into the [data_processing](https://github.com/Ivan-Zhou/Mozilla_UofT_Capstone_User_Feedback/blob/master/data_processing.py) python file. By runing the file, it will automatically extract data from the input folder and run the pipeline. 

### Output
Output data will be saved in a SQLite database in the Output folder (which does not required to be created manually). If no database is found (as in the current case), a new database file will be automatically created in the target folder. All the results from the later jobs will be automatically updated in the database.

<br>

## Specification and Extension of Input Data
Not all of our data sources have the same format in columns and data structure; therefore, users need to provide specification of the input data. 

In this tool we have included the data specification for the following type of datasets. We will use them as examples to help users understand the data specification and extend the tool for new data sources. 
- iOS feedback data from Appbot
- iOS feedback data from SurveyGizmo
- Desktop feedback data from SurveyGizmo

### Specificaiton of Column Names
First, users need to map the column names in the input data with the required column names so that the system can know where to extract the right information. 
Here we already taken into account that not every column is available in all the datasets, and most of the column requires pre-processing. Therefore,
the specification will provide inputs to the processing functions, instead of hardcode it in the system. The complete list of column names including:
- `Device`: the name of the column that contains the device information, aka "iPhone", "Windows"
- `Store`: the name of the column that contains the store information, aka "iOS", "Desktop"
- `Country`: the name of the column that contains the country information
- `Date`: the name of the column that contains the review date information
- `Rating`: the name of the column that contains the review rating information
- `Version`: the name of the column that contains the App Version information
- `Original Reviews`: the name of the column that contains the reviews texts

As examples, the specification of column names in the SurveyGizmo and Appbot datasets can be found in [SurveyGizmo](/spec/input_data_columns/survey_gizmo.py) and [Appbot](/spec/input_data_columns/appbot.py)


### Specification of Device Types and Store 
Each data source may provide us feedbacks for more than one device type and stores. For example, SurveyGizmo gives us feedbacks for iPhone, iPad, 
