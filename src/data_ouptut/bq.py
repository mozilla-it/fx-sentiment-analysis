import datetime
from src.support.support_functions import *
from src.data_ouptut.output_processing import filter_by_date
from google.cloud import bigquery
import logging

bq = bigquery.Client()
logger = logging.getLogger(__name__)

def _insert_bq(table,stuff):
    if len(stuff) == 0:
        return
    table_ref = bq.dataset("sentiments").table(table)
    _table = bq.get_table(table_ref)
    logger.debug(bq.insert_rows(_table,stuff))

def load_into_database(df_reviews, df_categorization, df_key_issue):
    max_date = get_max_date()
    logger.debug(max_date)
    df_reviews, df_categorization, df_key_issue = filter_by_date(df_reviews, df_categorization, df_key_issue, max_date)
    initial_id = initiate_id()
    insert_review_list(initial_id, df_reviews)
    insert_categorization_list(initial_id, df_categorization)
    insert_key_issue_list(initial_id, df_key_issue)

def get_max_date():
    """
    Get the latest review dates in the database - we will not cover the previous loaded one if there are overlap
    :return: a date
    """
    sql = ''' SELECT MAX(Review_Date) AS review_date from sentiments.reviews'''
    query_job = bq.query(sql)
    results = list(query_job.result())
    if len(results) > 0:
      extraction = results[0].review_date
      max_date = extraction.strftime('%Y-%m-%d') if extraction else '1999-01-01'
    else:
      max_date = '1999-01-01'
    return max_date

def insert_review_list(initial_id, df_reviews):
    idx = df_reviews["Translated Reviews"].notnull()
    df = df_reviews[idx]
    logger.info("# reviews to insert: ")
    logger.info(len(df))

    df["Original Reviews"] = df["Original Reviews"].str[:4000]
    df["Translated Reviews"] = df["Translated Reviews"].str[:4000]
    df["ID"] += initial_id
    df["ID"] = df["ID"].astype(str)
    df['clear_filters'] = 'Clear Filters'
    df['Spam'] = df['Spam'].astype(int)
    df['Translated Reviews'] = df['Translated Reviews'].replace('|',' ')
    df['Original Reviews'] = df['Original Reviews'].replace('|',' ')

    reviews = []
    for i, row in df.iterrows():
        date_value = row['Date'].strftime('%Y-%m-%d')
        reviews += [(row["ID"], row["Store"], row["Device"], row["Source"], row["Country"], date_value, row["Version"], row["Rating"],row["Original Reviews"],row["Translated Reviews"],row["Sentiment"],row["Spam"],row["Verb Phrases"],row["Noun Phrases"],row["clear_filters"])]

    _insert_bq("reviews",reviews)

def insert_categorization_list(initial_id, df_categorization):
    cats = []
    for i, row in df_categorization.iterrows():
        id_value = str(row['ID'] + initial_id)
        feature_value = row['Feature']
        component_value = row['Component']
        action_value = ', '.join(map(str, row['Actions']))
        cats += [(id_value, feature_value, component_value, action_value)]
    _insert_bq("categorization",cats)

def insert_key_issue_list(initial_id, df_key_issue):
    issues = []
    for i, row in df_key_issue.iterrows():
        id_value = str(row['ID'] + initial_id)
        key_issue_value = row['Issue']
        issues += [(id_value, key_issue_value)]
    _insert_bq("key_issue",issues)

def select_max_id_from_table(table):
    """
    Select the MAX ID from the existing table
    :param table:
    :return:
    """
    sql = "SELECT COALESCE(CAST(CAST(MAX(ID) AS FLOAT64) AS INT64),0) AS max_id from {}".format(table)
    query_job = bq.query(sql)
    max_id = list(query_job.result())[0].max_id
    return max_id

def initiate_id():
    now = datetime.now()
    current_date = int('%04g%02g%02g' % (now.year,now.month,now.day))
    initial_id = int(str(current_date) + '000000')  # Default value

    # Check if a set of data has already been imported today
    max_id = max([
        select_max_id_from_table('sentiments.reviews'),
        select_max_id_from_table('sentiments.categorization'),
        select_max_id_from_table('sentiments.key_issue'),
        ])
    if max_id > 0:
        if int(str(max_id)[:8]) == current_date:
            initial_id = max_id+1

    return initial_id

