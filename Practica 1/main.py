# -*- coding: utf-8 -*-

"""

{Description}
{License_info}

"""

__author__ = 'Leonardo'
__copyright__ = 'Copyright 2019, Dota 2 Team Composition Association'
__credits__ = ['Leonardo Henrique da Rocha Araujo']
__license__ = 'GNU GLPv3'
__version__ = '0.1.0'
__maintainer__ = 'Leonardo'
__email__ = 'leonardo.araujo@isistan.unicen.edu.ar'
__status__ = 'Dev'

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import json
import csv

def split_database(path):
	base_file_name = './dota2Dataset/dota2-'
	file_names = []
	files = {}
	
	for i in range(0, 23):
		for j in range(-1, 8):
			file_names.append('{}_{}'.format(i, j))
			files[file_names[-1]] = []
	
	with open(path, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar="'")
		
		for row in reader:
			gm = row[2]
			gt = row[3]
			
			if(gm.isdigit() and gt.isdigit()):
				files['{}_{}'.format(gm, gt)].append(row)
			
	for f_name in file_names:
		lines_len = len(files[f_name])
		header = Team Victory, Cluster ID, Game Mode, Game Type, Champion 1, Champion 2, Champion 3, Champion 4, Champion 5, Champion 6, Champion 7, Champion 8, Champion 9, Champion 10, Champion 11, Champion 12, Champion 13, Champion 14, Champion 15, Champion 16, Champion 17, Champion 18, Champion 19, Champion 20, Champion 21, Champion 22, Champion 23, Champion 24, Champion 25, Champion 26, Champion 27, Champion 28, Champion 29, Champion 30, Champion 31, Champion 32, Champion 33, Champion 34, Champion 35, Champion 36, Champion 37, Champion 38, Champion 39, Champion 40, Champion 41, Champion 42, Champion 43, Champion 44, Champion 45, Champion 46, Champion 47, Champion 48, Champion 49, Champion 50, Champion 51, Champion 52, Champion 53, Champion 54, Champion 55, Champion 56, Champion 57, Champion 58, Champion 59, Champion 60, Champion 61, Champion 62, Champion 63, Champion 64, Champion 65, Champion 66, Champion 67, Champion 68, Champion 69, Champion 70, Champion 71, Champion 72, Champion 73, Champion 74, Champion 75, Champion 76, Champion 77, Champion 78, Champion 79, Champion 80, Champion 81, Champion 82, Champion 83, Champion 84, Champion 85, Champion 86, Champion 87, Champion 88, Champion 89, Champion 90, Champion 91, Champion 92, Champion 93, Champion 94, Champion 95, Champion 96, Champion 97, Champion 98, Champion 99, Champion 100, Champion 101, Champion 102, Champion 103, Champion 104, Champion 105, Champion 106, Champion 107, Champion 108, Champion 109, Champion 110, Champion 111, Champion 112, champion 113
		
		if(lines_len > 0):
			with open(base_file_name + f_name + '.csv', 'w', newline='') as csvfile:
				writer = csv.writer(csvfile, delimiter=',', quotechar="'")
				writer.writerow([header])
				
				for row in files[f_name]:
					writer.writerow(row)

def get_team_comp_data(data_frame, filters = []):
	heroes = ['']*113
	heroes_file = open('./dota2Dataset/heroes.json', 'r')
	json_cont = json.load(heroes_file)
	
	for element in json_cont['heroes']:
		heroes[element['id'] - 1] = element['localized_name'].replace(' ', '_').replace("'", '')
	
	new_data = []
	
	rows_num = data_frame.shape[0]
	perc = 0
	last_perc = -1
	
	for index, row in data_frame.iterrows():
		perc = index*100.0/rows_num
		
		if(int(perc)%10 == 0 and last_perc != int(perc)):
			print("{}%".format(perc))
			last_perc = int(perc)
		
		new_row_t1 = []
		new_row_t2 = []
		victory_t1 = "Victory" if row['Team Victory'] == 1 else "Defeat"
		victory_t2 = "Victory" if row['Team Victory'] == -1 else "Defeat"
		
		new_row_t1.append(victory_t1)
		new_row_t2.append(victory_t2)
		
		for i in range(113):
			
			if(row.iloc[4 + i] == 1):
				new_row_t1.append(heroes[i])
			elif(row.iloc[4 + i] == -1):
				new_row_t2.append(heroes[i])
		
		new_data.append(new_row_t1)
		new_data.append(new_row_t2)
		
	return new_data
	
def frequent_items(dataset, support=0.1):
	te = TransactionEncoder()
	te_ary = te.fit(dataset).transform(dataset)
	df = pd.DataFrame(te_ary, columns=te.columns_)
	frequent_itemsets = apriori(df, min_support=support, use_colnames=True)
	
	return frequent_itemsets
	
def assoc_rul(frequent_itemsets, thresh):
	return association_rules(frequent_itemsets, metric="confidence", min_threshold=thresh)

def main():
	print()
	data_df = pd.read_csv("./dota2Dataset/dota2Full.csv", index_col=False)
	new_data = get_team_comp_data(data_df)
	
if __name__ == "__main__":
    main()
