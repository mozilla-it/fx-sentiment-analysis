import sqlite3
from sqlite3 import Error
from src.support_functions import *
from src.print_out_results import print_contents

db = "reviews.sqlite"
table_feedbacks = 'Reviews'


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
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
                                                    ID text PRIMARY KEY,
                                                    Store text NOT NULL,
                                                    Source text NOT NULL,
                                                    Country text,
                                                    'Review Date' text NOT NULL,
                                                    Version FLOAT NOT NULL,
                                                    Rating integer,
                                                    'Original Reviews' text NOT NULL,
                                                    'Translated Reviews' text NOT NULL,
                                                    Sentiment text NOT NULL,
                                                    Spam integer,
                                                    'Verb Phrases' text,
                                                    'Noun Phrases' text,
                                                    'Clear Filters'
                                                ); """
        return sql_create_reviews_table

    def get_sql_create_categorization_table():
        sql_create_categorization_table = """ CREATE TABLE IF NOT EXISTS categorization (
                                                    ID text NOT NULL,
                                                    Feature text NOT NULL,
                                                    Component text NOT NULL,
                                                    theAction text NOT NULL
                                                ); """
        return sql_create_categorization_table

    def get_sql_create_key_issue_table():
        sql_create_categorization_table = """ CREATE TABLE IF NOT EXISTS key_issue (
                                                    ID text NOT NULL,
                                                    'Key Issue' NOT NULL
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
        except Error as e:
            print(e)

    sql_create_reviews_table = get_sql_create_reviews_table()
    sql_create_categorization_table = get_sql_create_categorization_table()
    sql_create_key_issue_table = get_sql_create_key_issue_table()
    create_table(conn, sql_create_reviews_table)
    create_table(conn, sql_create_categorization_table)
    create_table(conn, sql_create_key_issue_table)


def insert_review(conn, review):
    sql = ''' INSERT INTO reviews(ID, Store, Source, Country, 'Review Date', Version, Rating, 'Original Reviews', 
              'Translated Reviews', Sentiment, Spam, 'Verb Phrases', 'Noun Phrases', 'Clear Filters')
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, review)
    return cur.lastrowid


def insert_categorization(conn, cate):
    sql = ''' INSERT INTO categorization(ID, Feature, Component, theAction)
              VALUES(?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, cate)
    return cur.lastrowid


def insert_key_issue(conn, key_issue):
    sql = ''' INSERT INTO key_issue(ID, 'Key Issue')
              VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, key_issue)
    return cur.lastrowid


def insert_review_list(conn, initial_id, file_path):
    df_reviews_path = file_path + 'feedbacks.csv'
    df_reviews = pd.read_csv(df_reviews_path)
    for i, row in df_reviews.iterrows():
        if not isNaN(row['Translated Reviews']):
            id_value = str(row['ID'] + initial_id)
            store_value = row['Store']
            source_value = row['Source']
            country_value = row['Country']
            date_value = row['Date']
            version_value = row['Version']
            rating_value = row['Rating']
            original_reviews_value = row['Original Reviews']
            translated_reviews_value = row['Translated Reviews']
            sentiment_value = row['Sentiment']
            spam_value = int(row['Spam'])
            verb_phrases_value = row['Verb Phrases']
            noun_phrases_value = row['Noun Phrases']
            clear_filters = 'Clear Filters'
            review = (id_value, store_value, source_value, country_value, date_value, version_value, rating_value,
                          original_reviews_value, translated_reviews_value, sentiment_value, spam_value,
                          verb_phrases_value, noun_phrases_value, clear_filters)
            insert_review(conn, review)


def insert_categorization_list(conn, initial_id, file_path):
    df_cate_path = file_path + 'categorization.csv'
    df_cate = pd.read_csv(df_cate_path)
    for i, row in df_cate.iterrows():
        id_value = str(row['ID'] + initial_id)
        feature_value = row['Feature']
        component_value = row['Component']
        action_value = row['Actions']
        new_cate = (id_value, feature_value, component_value, action_value)
        insert_categorization(conn, new_cate)


def insert_key_issue_list(conn, initial_id, file_path):
    df_key_issue_path = file_path + 'key_issue.csv'
    df_key_issue = pd.read_csv(df_key_issue_path)
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

        sql = ''' SELECT MAX(CAST(ID AS Int)) from key_issue'''
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


def update_db(conn, data_file_path):
    check_tables(conn)
    initial_id = initiate_id(conn)

    insert_review_list(conn, initial_id, data_file_path)
    insert_categorization_list(conn, initial_id, data_file_path)
    insert_key_issue_list(conn, initial_id, data_file_path)


def remove_db(db):
    import os
    os.remove(db)


def main(db, files_to_be_read):
    # create a database connection
    conn = create_connection(db)

    if conn is not None:
        for file_path in files_to_be_read:
            update_db(conn, file_path)
    else:
        print('Error! Cannot create the database connection!')
    conn.commit()
    conn.close()


def read_db():
    # create a database connection
    conn = create_connection(db)

    if conn is not None:
        # select_all_from_table(conn, 'reviews')
        # select_all_from_table(conn, 'categorization')
        # select_all_from_table(conn, 'key_issue')
        print_contents_from_db(conn, 'reviews', 'categorization', 'key_issue')
    else:
        print('Error! Cannot create the database connection!')
    conn.close()


def print_contents_from_db(conn, df_reviews_name, df_cate_name, df_key_issue_name):
    df_reviews = pd.read_sql_query("SELECT * FROM " + df_reviews_name, conn)
    df_reviews['ID'] = df_reviews['ID'].astype(int)
    df_categorization = pd.read_sql_query("SELECT * FROM " + df_cate_name, conn)
    df_categorization['ID'] = df_categorization['ID'].astype(int)
    df_key_issue = pd.read_sql_query("SELECT * FROM " + df_key_issue_name, conn)
    df_key_issue['ID'] = df_key_issue['ID'].astype(int)
    print_contents(df_categorization, df_reviews, df_key_issue)


"""
if __name__ == '__main__':
    db = "reviews.sqlite"
    files_to_be_read = ['Data/2018_02_22/output/']
    remove_db(db)
    main(db, files_to_be_read)
    read_db()
"""