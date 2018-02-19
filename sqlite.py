import sqlite3
from sqlite3 import Error
import pandas as pd


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


def insert_review(conn, review):
    sql = ''' INSERT INTO reviews(ID, Store, Source, Country, 'Review Date', Version, Rating, 'Original Reviews', 
              'Translated Reviews', Sentiment, Spam, 'Verb Phrases', 'Noun Phrases', 'Clear Filters')
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, review)
    return cur.lastrowid


def insert_categorization(conn, cate):
    sql = ''' INSERT INTO categorization(ID, theAction, Component, Feature)
              VALUES(?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, cate)
    return cur.lastrowid


def insert_tag(conn, tag):
    sql = ''' INSERT INTO tag(ID, Component, Tag)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, tag)
    return cur.lastrowid


def insert_review_list(conn):
    df_reviews_path = 'Data/test_2/output/feedbacks.csv'
    df_reviews = pd.read_csv(df_reviews_path)
    countries = ['USA', 'Canada', 'China', 'Singapore', 'Mexico']
    for i, row in df_reviews.iterrows():
        id_value = row['ID']
        store_value = row['Store']
        source_value = row['Source']
        country_value = countries[i % len(countries)]
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


def insert_categorization_list(conn):
    df_cate_path = 'Data/test_2/output/categorization.csv'
    df_cate = pd.read_csv(df_cate_path)
    for i, row in df_cate.iterrows():
        action_value = row['Action']
        component_value = row['Component']
        feature_value = row['Feature']
        id_value = row['ID']
        new_cate = (id_value, action_value, component_value, feature_value)
        insert_categorization(conn, new_cate)


def insert_tag_list(conn):
    df_tag_path = 'Data/test_2/output/tag.csv'
    df_tag = pd.read_csv(df_tag_path)
    for i, row in df_tag.iterrows():
        component_value = row['Component']
        id_value = row['ID']
        tag_value = row['Tag']
        new_tag = (id_value, component_value, tag_value)
        insert_tag(conn, new_tag)


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


def get_sql_create_reviews_table():
    sql_create_reviews_table = """ CREATE TABLE IF NOT EXISTS reviews (
                                                ID integer PRIMARY KEY,
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
                                                ID integer NOT NULL,
                                                Feature text NOT NULL,
                                                Component text NOT NULL,
                                                theAction text NOT NULL
                                            ); """
    return sql_create_categorization_table


def get_sql_create_tag_table():
    sql_create_categorization_table = """ CREATE TABLE IF NOT EXISTS tag (
                                                ID integer NOT NULL,
                                                Component text NOT NULL,
                                                Tag NOT NULL
                                            ); """
    return sql_create_categorization_table


def delete_all_from_table(conn, table):
    """
    Delete all rows in the tasks table
    :param conn: Connection to the SQLite database
    :return:
    """
    sql = 'DELETE FROM ' + table
    cur = conn.cursor()
    cur.execute(sql)

def main():
    db = "reviews.sqlite"
    import os
    os.remove(db)

    # create a database connection
    conn = create_connection(db)
    sql_create_reviews_table = get_sql_create_reviews_table()
    sql_create_categorization_table = get_sql_create_categorization_table()
    sql_create_tag_table = get_sql_create_tag_table()

    if conn is not None:
        create_table(conn, sql_create_reviews_table)
        create_table(conn, sql_create_categorization_table)
        create_table(conn, sql_create_tag_table)
        delete_all_from_table(conn, 'reviews')
        delete_all_from_table(conn, 'categorization')
        delete_all_from_table(conn, 'tag')
        insert_review_list(conn)
        insert_categorization_list(conn)
        insert_tag_list(conn)
    else:
        print('Error! Cannot create the database connection!')
    conn.commit()
    conn.close()


def read_data():
    db = "reviews.sqlite"

    # create a database connection
    conn = create_connection(db)

    if conn is not None:
        select_all_from_table(conn, 'reviews')
        select_all_from_table(conn, 'categorization')
        select_all_from_table(conn, 'tag')
    else:
        print('Error! Cannot create the database connection!')
    conn.close()


if __name__ == '__main__':
    # main()
    read_data()