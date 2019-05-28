# -*- coding: utf-8 -*-

"""

{Description}
{License_info}

"""

__author__ = 'Leonardo'
__copyright__ = 'Copyright 2019, Poker advisor'
__credits__ = ['Leonardo Henrique da Rocha Araujo']
__license__ = 'GNU GLPv3'
__version__ = '0.1.0'
__maintainer__ = 'Leonardo'
__email__ = 'leonardo.araujo@isistan.unicen.edu.ar'
__status__ = 'Dev'

import pandas as pd
import numpy as np
import csv
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def read_dataset(path):
	att = []
	targ = []
	
	with open(path, 'r') as file:
		readCSV = csv.reader(file, delimiter=',')
		for row in readCSV:
			att.append([int(num) for num in row[:-1]])
			targ.append([int(row[-1])])
	
	np_att = np.array(att)
	np_targ = np.array(targ)
	
	return [np_att, np_targ]

def train_SVC(data_train):
	now = time.time()

	clf = svm.SVC(gamma=0.001, C=100.)
	clf.fit(data_train[0], data_train[1].ravel())  
	
	then = time.time()
	print('Training time ellapsed: {}'.format(then - now))
	
	return clf
	
def test_SVC(svc, data_test):
	now = time.time()
	
	data_predict = svc.predict(data_test[0])
	print('Accuracy:',metrics.accuracy_score(data_test[1], data_predict))
	
	then = time.time()
	print('Testing time ellapsed: {}'.format(then - now))
	
def train_knn(data_train, n):
	now = time.time()
	
	knn = KNeighborsClassifier(n_neighbors=n)
	knn.fit(data_train[0], data_train[1].ravel())
	
	then = time.time()
	print('Training time ellapsed: {}'.format(then - now))
	
	return knn

def test_knn(knn, data_test):
	now = time.time()
	
	data_predict = knn.predict(data_test[0])
	print('Accuracy:',metrics.accuracy_score(data_test[1], data_predict))
	
	then = time.time()
	print('Testing time ellapsed: {}'.format(then - now))
	
def predict_knn(value):
	now = time.time()
	
	pred = knn.predict(value)
	
	then = time.time()
	print('Predicting time ellapsed: {}'.format(then - now))
	
	return pred

def pre_process_dataset(path):
	with open(path, 'r') as file:
		new_lines = []
		new_targs = []
		readCSV = csv.reader(file, delimiter=',')
		
		for row in readCSV:
			att = [int(num) for num in row[:-1]]
			targ = int(row[-1])
			new_att = [0 for i in range(52)]
			
			for i in range(0, 10, 2):
				new_att[(att[i+1] + ((att[i] - 1) * 13)) - 1] = 1
			
			new_lines.append(new_att)
			new_targs.append(targ)
		
		np_att = np.array(new_lines)
		np_targ = np.array(new_targs)
	
	return [np_att, np_targ]

def save_dataset(dataset, path):
	with open(path, 'w', newline='') as file:
		writeCSV = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		total_lines = len(dataset[0])
		perc = 0
		last_perc = -1
		
		for i in range(total_lines):
			perc = i*100.0/total_lines
			
			if(int(perc)%5 == 0 and last_perc != int(perc)):
				print("{}%".format(perc))
				last_perc = int(perc)
			
			line = dataset[0][i].tolist()
			line.append(dataset[1][i])
			writeCSV.writerow(line)
	
def main():
	#path_train = './Poker Dataset/poker-hand-training-true.data'
	#path_test = './Poker Dataset/poker-hand-testing.data'
	
	#data_train = read_dataset(path_train)
	#data_test = read_dataset(path_test)
	
	#path_full = './Poker Dataset/poker-hand-full-changed.data'
	#data_full = read_dataset(path_full)
	
	path_train = './Poker Dataset/poker-hand-training-changed.data'
	path_test = './Poker Dataset/poker-hand-testing-changed500.data'
	
	data_train = read_dataset(path_train)
	data_test = read_dataset(path_test)
	
	trained_model_knn = train_knn(data_train, 7)
	test_knn(trained_model_knn, data_test)
	
	#svc = train_SVC(data_train)
	#test_SVC(svc, data_test)
	
	'''k_list = list(range(1, 50, 2))
	cv_scores = []
	
	x_train, y_train, x_test, y_test = train_test_split(data_full[0],data_full[1],test_size=0.1,random_state=123)
	
	for k in k_list:
		now = time.time()
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn, X=data_full[0], y=data_full[1].ravel(), scoring='accuracy', cv=5)
		then = time.time()
		print('It took: {} \nFor k={} \nAccuracy:{}'.format(then - now, k, scores.mean()))
		cv_scores.append(scores.mean())
		
	print(cv_scores)'''
	
	#for i in range(1, 20):
	#	print('{} nearest neighbors'.format(i))
	#	trained_model_knn = train_knn(data_train, i)
	#	test_knn(trained_model_knn, data_test)
	
	
if __name__ == "__main__":
    main()
