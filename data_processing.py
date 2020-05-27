#!/usr/bin/python3

from src.input.read_data import read_all_data
from src.pre_processing.preprocessing import preprocess
from src.categorization.categorization import categorize
from src.key_issues.summarization import cluster_and_summarize
from src.data_ouptut.bq import load_into_database
from src.extract.appbot import get_appbot_csv
from src.extract import get_csv_from_url
import pandas as pd
import os
import logging

APPBOT_USERNAME = os.environ.get('APPBOT_USERNAME','')
APPBOT_PASSWORD = os.environ.get('APPBOT_PASSWORD','')
SURVEYGIZMO_URL = os.environ.get('SURVEYGIZMO_URL','')

logging.basicConfig(level=logging.INFO)

def main():
  # Commenting this out because the Appbot account we have ("nawong@mozilla.com")'s trial
  # period has expired and cannot view any data.
  #
  #logging.info("Copying Appbot file")
  #get_appbot_csv(APPBOT_USERNAME,APPBOT_PASSWORD,'/workspace/Input/appbot.csv')

  logging.info("Copying SurveyGizmo file")
  get_csv_from_url(SURVEYGIZMO_URL,'/workspace/Input/surveygizmo.csv')

  df = read_all_data('/workspace/Input')
  df = preprocess(df)
  df_categorization, df = categorize(df)
  df_key_issue = cluster_and_summarize(df)
  load_into_database(df, df_categorization, df_key_issue)

if __name__ == '__main__':
  main()
