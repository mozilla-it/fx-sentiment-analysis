import numpy as np
import pandas as pd
from support_functions import *

data_path = 'Data/2017_11_17/output_py.xlsx'

read_over = False

if read_over:
    df = read_exist_output(data_path)
    df = categorize(df)
    df.to_csv('temp.csv')
else:
    df = pd.read_csv('temp.csv')
    df = df.replace(np.nan, '', regex=True)

feedbacks = df['Translated Reviews'].as_matrix()
similarity_matrix = measure_sim_tfidf(feedbacks, viz = False)

thresh = 0.3
indice_list = []
for index, x in np.ndenumerate(np.triu(similarity_matrix, k=1)):
    if x >= thresh:
        indice_list.append(index)

repeat = True
while repeat:
    list_length = sum([len(list) for list in indice_list])
    for indice1 in indice_list:
        for indice2 in indice_list:
            if not indice1 == indice2:
                if bool(set(indice1) & set(indice2)):
                    merged_list = set(indice1)|set(indice2)
                    if merged_list not in indice_list:
                        indice_list.append(merged_list)
                    indice_list.remove(indice2)
    if list_length == sum([len(list) for list in indice_list]): # Stabilize
        repeat = False

print(max([len(list) for list in indice_list]))
print(list_length)
print(len(df))
print(len(indice_list))

for i in range(len(indice_list)):
    print('Group' + str(i))
    for index in indice_list[len(indice_list)-i-1]:
        print(feedbacks[index])
        print()
    print()
    print()