
import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, tree 
from sklearn import tree
from sklearn.externals.six import StringIO 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

import pydot
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import itertools


import logging
import os
from functools import partial
from multiprocessing.pool import Pool
from time import time, sleep
import time as timer
import queue 
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent.futures
 
import util

 
    
def createDT(data):
    dt = DTRun(data) 
    t = threading.Thread(target=dt.createParams)
    t.start() 
    timer.sleep(2)
    return dt 


def createEntry(dataType, modelType, clf):
    
    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
 

    scoreList = util.ScoreList('Learners')

    start = time()
    clf.fit(xTrain, yTrain)
    end = time()
    trainTime = end - start

    start = time()
    testPred = clf.predict(xTest)
    end = time()

    queryTime = end - start

    trainPred = clf.predict(xTrain)
    score = scoreList.Add(yTest, testPred, yTrain, trainPred, 'abd')
    return [dataType, modelType, score.Accuracy, score.Precision, score.Recall, score.F1, trainTime, queryTime]

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
def runAll(dataType):
 
    package = data.createData(dataType)
    timedScores = []
    if (dataType == 'Heart'):
  
        params = {'class_weight': None, 'criterion': 'entropy', 'max_features': None, 'min_samples_leaf': 10, 'splitter': 'best'}
        clf = DecisionTreeClassifier(random_state=util.randState) 
        clf.set_params(**params)
        clf.max_depth = 8  
        clf.min_samples_leaf = 10 
        timedScores.append(createEntry(dataType, 'Decision Tree', clf))

        clf = AdaBoostClassifier()  
        clf.n_estimators = 150
        timedScores.append(createEntry(dataType, 'Boosted', clf))
    
        params = {'activation': 'tanh', 'learning_rate': 'constant', 'solver': 'adam'}
        input = package.features.shape[1]
        input = int(.7 * input) 
        clf = MLPClassifier(hidden_layer_sizes = (input,5,2)) 
        clf.set_params(**params)   
        clf.max_iter = 600
        timedScores.append(createEntry(dataType, 'Neural Network', clf))
 
        clf = SVC(cache_size=5000, max_iter=5000, C = .5, gamma = 1, kernel='rbf')  
        timedScores.append(createEntry(dataType, 'SVM', clf))
  
        params = {'algorithm': 'ball_tree', 'p': 1, 'weights': 'distance'} 
        clf = KNeighborsClassifier()
        clf.set_params(**params)    
        clf.n_neighbors = 12
        timedScores.append(createEntry(dataType, 'KNN', clf))
 
    else:
        params = {'class_weight': None, 'criterion': 'entropy', 'max_features': None, 'min_samples_leaf': 50, 'splitter': 'best'} 
        clf = DecisionTreeClassifier(random_state=util.randState) 
        clf.set_params(**params)
        clf.max_depth = 5  
        clf.min_samples_leaf = 50
        timedScores.append(createEntry(dataType, 'Decision Tree', clf))
 
        clf = AdaBoostClassifier()  
        clf.n_estimators = 40
        timedScores.append(createEntry(dataType, 'Boosted', clf))

        params = {'activation': 'relu', 'learning_rate': 'invscaling', 'solver': 'lbfgs'} 
        
        input = package.features.shape[1]
        input = int(.7 * input) 
        clf = MLPClassifier(hidden_layer_sizes = (input,7,2)) 
        clf.max_iter = 150
        clf.set_params(**params)    
        timedScores.append(createEntry(dataType, 'Neural Network', clf))
        
        clf = SVC(cache_size=5000, max_iter=5000, C = 10, gamma = 0.01, kernel='rbf')  
        timedScores.append(createEntry(dataType, 'SVM', clf))
            
        params = {'algorithm': 'auto', 'p': 1, 'weights': 'distance'}  
        clf = KNeighborsClassifier()
        clf.set_params(**params)    
        clf.n_neighbors = 5 
        timedScores.append(createEntry(dataType, 'KNN', clf))

    
    
    return timedScores


 
import dt
import network
import knn
import svm
import ensemble
import boosted
import plotter
import data



if __name__ == '__main__':
    
    plotter.showPlot = False
    # plotter.showPlot = False
    data.rows = 1000
    data.rows = 10000
    data.rows = 100000
    data.rows = 5000
    # data.rows = 1000000

    startTime = time()
    np.set_printoptions(suppress=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    dataType = 'los'
    dataType = 'heart'
    dataType = 'adult'
    
    dataTypes = ['los', 'heart', 'adult']
    dataTypes = ['los', 'heart', 'adult']
    dataTypes = [ 'Heart', 'Adult']
    # dataTypes = ['heart']
    # dataTypes = ['adult']
    scores = []
    for dataType in dataTypes:
        print(dataType)

        scores.append(runAll(dataType))
        
        print('DT')
        dt.run(dataType)
        
        print('Neural Network')
        network.run(dataType)

        print('Boosted')
        boosted.run(dataType)
        
        
        print('SVM')
        svm.run(dataType)
        
        
        print('KNN')
        knn.run(dataType)
    endTime = time()
    print(str(endTime - startTime))

    print(scores)
 