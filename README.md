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
Output data will be saved in a SQLite database in the [Output](/Output) folder. If no database is found (as in the current case), a new database file will be automatically created in the target folder. All the results from the later jobs will be automatically updated in the database.
