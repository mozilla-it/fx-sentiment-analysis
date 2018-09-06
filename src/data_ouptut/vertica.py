#import sqlite3
#from sqlite3 import Error
import datetime
from src.support.support_functions import *
from src.data_ouptut.output_processing import filter_by_date
import vertica_python

#output_path = 'Output/'
#db = output_path + "reviews.sqlite"


def load_into_database(df_reviews, df_categorization, df_key_issue,user,pwd,host,db):
    # create a database connection
    conn_info={'host': host,'port': 5433,'user': user,'password': pwd,'database': db,'read_timeout': 6000,'unicode_error': 'strict','ssl': False,'connection_timeout': 5}
    conn = create_connection(conn_info)

    if conn is not None:
        check_tables(conn)
        max_date = get_max_date(conn)
        df_reviews, df_categorization, df_key_issue = filter_by_date(
            df_reviews, df_categorization, df_key_issue, max_date)
        initial_id = initiate_id(conn)
        insert_review_list_vertica(conn, initial_id, df_reviews)
        insert_categorization_list(conn, initial_id, df_categorization)
        insert_key_issue_list(conn, initial_id, df_key_issue)
    else:
        print('Error! Cannot create the database connection!')
    conn.commit()
    conn.close()


def get_max_date(conn):
    """
    Get the latest review dates in the database - we will not cover the previous loaded one if there are overlap
    :param conn:
    :return: a date
    """
    sql = ''' SELECT MAX("Review Date") from reviews'''
    cur = conn.cursor()
    cur.execute(sql)
    extraction = cur.fetchone()[0]
    max_date = extraction.strftime('%Y-%m-%d') if extraction else '1999-01-01'
    return max_date


def create_connection(params):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = vertica_python.connect(**params)
        return conn
    except Exception as e:
        print(e)
    return None


def check_tables(conn):
    """
    Check if all the tables exist
    :param conn:
    :return:
    """

    def get_sql_create_reviews_table():
        sql_create_reviews_table = """ CREATE TABLE IF NOT EXISTS reviews (
                                                    ID VARCHAR PRIMARY KEY,
                                                    Store VARCHAR NOT NULL,
                                                    Device VARCHAR NOT NULL,
                                                    Source VARCHAR NOT NULL,
                                                    Country VARCHAR,
                                                    "Review Date" DATE NOT NULL,
                                                    Version FLOAT NOT NULL,
                                                    Rating VARCHAR,
                                                    "Original Reviews" VARCHAR(4000) NOT NULL,
                                                    "Translated Reviews" VARCHAR(4000) NOT NULL,
                                                    Sentiment VARCHAR NOT NULL,
                                                    Spam integer,
                                                    "Verb Phrases" VARCHAR(4000),
                                                    "Noun Phrases" VARCHAR(4000),
                                                    "Clear Filters" VARCHAR
                                                ); """
        return sql_create_reviews_table

    def get_sql_create_categorization_table():
        sql_create_categorization_table = """ CREATE TABLE IF NOT EXISTS categorization (
                                                    ID VARCHAR NOT NULL,
                                                    Feature VARCHAR NOT NULL,
                                                    Component VARCHAR NOT NULL,
                                                    theAction VARCHAR(4000)
                                                ); """
        return sql_create_categorization_table

    def get_sql_create_key_issue_table():
        sql_create_categorization_table = """ CREATE TABLE IF NOT EXISTS key_issue (
                                                    ID VARCHAR NOT NULL,
                                                    "Key Issue" VARCHAR NOT NULL
                                                ); """
        return sql_create_categorization_table

    def create_table(conn, create_table_sql):
        """ create a table from the create_table_sql statement
        :param conn: Connection object
        :param create_table_sql: a CREATE TABLE statement
        :return:
        """
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
        except Exception as e:
            print(e)

    sql_create_reviews_table = get_sql_create_reviews_table()
    sql_create_categorization_table = get_sql_create_categorization_table()
    sql_create_key_issue_table = get_sql_create_key_issue_table()
    create_table(conn, sql_create_reviews_table)
    create_table(conn, sql_create_categorization_table)
    create_table(conn, sql_create_key_issue_table)

    
def insert_review_list_vertica(conn, initial_id, df_reviews):
    def insert_review(conn, review):
        sql = ''' INSERT INTO reviews(ID, Store, Device, Source, Country, "Review Date", Version, Rating, "Original Reviews",
                  "Translated Reviews", Sentiment, Spam, "Verb Phrases", "Noun Phrases", "Clear Filters")
                  VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
        cur = conn.cursor()
        cur.execute(sql,list(review))

    df = df_reviews[df_reviews["Translated Reviews"].notnull()]
    print("# reviews to insert: ",len(df))

    df["Original Reviews"] = df["Original Reviews"].str[:4000]
    df["Translated Reviews"] = df["Translated Reviews"].str[:4000]
    df["ID"] += initial_id
    df["ID"] = df["ID"].astype(str)
    df['clear_filters'] = 'Clear Filters'
    df['Spam'] = df['Spam'].astype(int)
    df['Translated Reviews'] = df['Translated Reviews'].replace('|',' ')
    df['Original Reviews'] = df['Original Reviews'].replace('|',' ')

    for i, row in df.iterrows():
        date_value = row['Date'].strftime('%m-%d-%Y')
        review = (row["ID"], row["Store"], row["Device"], row["Source"], row["Country"], date_value, row["Version"], row["Rating"],row["Original Reviews"],row["Translated Reviews"],row["Sentiment"],row["Spam"],row["Verb Phrases"],row["Noun Phrases"],row["clear_filters"])
        insert_review(conn, review)

    
def insert_review_list(conn, initial_id, df_reviews):
    def insert_review(conn, review):
        sql = ''' INSERT INTO reviews(ID, Store, Device, Source, Country, "Review Date", Version, Rating, "Original Reviews", 
                  "Translated Reviews", Sentiment, Spam, "Verb Phrases", "Noun Phrases", "Clear Filters")
                  VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
        cur = conn.cursor()
        print(sql)
        print(review)
        cur.execute(sql,list(review))
        #return cur.lastrowid

    for i, row in df_reviews.iterrows():
        if not isNaN(row['Translated Reviews']):
            id_value = str(row['ID'] + initial_id)
            store_value = row['Store']
            device_value = row['Device']
            source_value = row['Source']
            country_value = row['Country']
            date_value = row['Date'].strftime('%m-%d-%Y')
            version_value = row['Version']
            rating_value = row['Rating']
            original_reviews_value = row['Original Reviews']
            translated_reviews_value = row['Translated Reviews']
            sentiment_value = row['Sentiment']
            spam_value = int(row['Spam'])
            verb_phrases_value = row['Verb Phrases']
            noun_phrases_value = row['Noun Phrases']
            clear_filters = 'Clear Filters'
            review = (id_value, store_value, device_value, source_value, country_value, date_value, version_value, rating_value,
                          original_reviews_value, translated_reviews_value, sentiment_value, spam_value,
                          verb_phrases_value, noun_phrases_value, clear_filters)
            
            insert_review(conn, review)


def insert_categorization_list(conn, initial_id, df_categorization):
    def insert_categorization(conn, cate):
        sql = ''' INSERT INTO categorization(ID, Feature, Component, theAction)
                  VALUES(%s,%s,%s,%s) '''
        cur = conn.cursor()
        cur.execute(sql, cate)
        #return cur.lastrowid

    for i, row in df_categorization.iterrows():
        id_value = str(row['ID'] + initial_id)
        feature_value = row['Feature']
        component_value = row['Component']
        action_value = ', '.join(map(str, row['Actions']))
        new_cate = (id_value, feature_value, component_value, action_value)
        insert_categorization(conn, new_cate)


def insert_key_issue_list(conn, initial_id, df_key_issue):
    def insert_key_issue(conn, key_issue):
        sql = ''' INSERT INTO key_issue(ID, "Key Issue")
                  VALUES(%s,%s) '''
        cur = conn.cursor()
        cur.execute(sql, key_issue)
        #return cur.lastrowid

    for i, row in df_key_issue.iterrows():
        id_value = str(row['ID'] + initial_id)
        key_issue_value = row['Issue']
        new_key_issue = (id_value, key_issue_value)
        insert_key_issue(conn, new_key_issue)


def select_all_from_table(conn, table):
    """
    Query all rows in the reviews table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM " + table)

    rows = cur.fetchall()
    print('Start reading: ')
    for row in rows:
        print(row)


def delete_all_from_table(conn, table):
    """
    Delete all rows in the tasks table
    :param conn: Connection to the SQLite database
    :return:
    """
    sql = 'DELETE FROM ' + table
    cur = conn.cursor()
    cur.execute(sql)


def initiate_id(conn):
    def get_id_date_part():
        """
        :return: an 8-digit number to represent the current date in integer type
        """
        now = datetime.now()
        year_str = str(now.year)
        month_str = str(now.month) if len(str(now.month)) > 1 else '0' + str(now.month)
        day_str = str(now.day) if len(str(now.day)) > 1 else '0' + str(now.day)
        date = int(year_str + month_str + day_str)
        return date

    def select_max_id_from_tables(conn):
        """
        Select the MAX ID from the existing table
        :param conn:
        :return:
        """
        max_id = 0
        sql = ''' SELECT MAX(CAST(ID AS Int)) from reviews'''
        cur = conn.cursor()
        cur.execute(sql)
        extraction = cur.fetchone()[0]
        max_result = extraction if extraction else 0
        max_id = max(max_id, max_result)

        sql = ''' SELECT MAX(CAST(ID AS Int)) from categorization'''
        cur = conn.cursor()
        cur.execute(sql)
        extraction = cur.fetchone()[0]
        max_result = extraction if extraction else 0
        max_id = max(max_id, max_result)

        sql = ''' SELECT MAX(to_number(ID)) from key_issue'''
        cur = conn.cursor()
        cur.execute(sql)
        extraction = cur.fetchone()[0]
        max_result = extraction if extraction else 0
        max_id = max(max_id, max_result)
        return max_id

    current_date = get_id_date_part()  # 8-digit date in integer type
    initial_id = int(str(current_date) + '000000')  # Default value

    # Check if a set of data has already been imported today
    max_id = select_max_id_from_tables(conn)  # Get the max of the existing ID
    if max_id > 0:
        if int(str(max_id)[:8]) == current_date:
            initial_id = max_id+1

    return initial_id

'''
def remove_db(db):
    import os
    os.remove(db)
'''

def extract_contents_from_db(params):
    conn = create_connection(params)
    df_reviews = pd.read_sql_query("SELECT * FROM reviews", conn)
    df_reviews['ID'] = df_reviews['ID'].astype(int)
    df_categorization = pd.read_sql_query("SELECT * FROM categorization", conn)
    df_categorization['ID'] = df_categorization['ID'].astype(int)
    df_key_issue = pd.read_sql_query("SELECT * FROM key_issue", conn)
    df_key_issue['ID'] = df_key_issue['ID'].astype(int)
    return df_reviews, df_categorization, df_key_issue
