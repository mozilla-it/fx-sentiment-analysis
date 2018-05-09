from src.input.read_data import read_all_data
from src.pre_processing.preprocessing import preprocess
from src.categorization.categorization import categorize
from src.key_issues.summarization import cluster_and_summarize
from src.data_ouptut.vertica import load_into_database
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-u','--username',
                    action='store',         
                    help='The Vertica username for authentication.')
parser.add_argument('-p','--password',
                    action='store',           
                    help='The Vertica password for authentication.')
parser.add_argument('--host',
                    action='store',
                    help='The Vertica server')
parser.add_argument('-db','--database',
                    action='store',
                    help='The Vertica database')
args = parser.parse_args()

if not args.username:
	print("Vertica username is missing")
elif not args.password:
	print("Vertica password is missing")
elif not args.host:
	print("The hostname is missing")
elif not args.database:
	print("The database is missing")
else:
	vu =args.username #Vertica Username
	vp = args.password #Vertica Password
	vdb = args.database
	vh = args.host
	def data_processing():
    		df = read_all_data()
    		df = preprocess(df)
    		df_categorization, df = categorize(df)
    		df_key_issue = cluster_and_summarize(df)
    		load_into_database(df, df_categorization, df_key_issue,vu,vp,vh,vdb)
		

	if __name__ == '__main__':
    		data_processing()

