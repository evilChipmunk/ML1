
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy import array
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, tree
from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import pydot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
 
import util
import data
import plotter



def createTree(depth, params):        
    clf = DecisionTreeClassifier(max_depth=depth, random_state=util.randState)
    clf.set_params(**params)
    return clf

import searcher

def heart(dataType):
    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
    title =  '{0} Decision Tree'.format(dataType)
    xLabel = 'Depth'
    scoreList = util.ScoreList(xLabel)
 
   
    param_range = list(range(1, 20))
    param = 'max_depth'
    params = {'class_weight': None, 'criterion': 'entropy', 'max_features': None, 'min_samples_leaf': 10, 'splitter': 'best'}
  
    clf_tree = DecisionTreeClassifier(random_state=util.randState) 
    clf_tree.set_params(**params)    
    plotter.plotValidationCurve(clf_tree, xTrain, yTrain, param, param_range, graphTitle=title + ' Max Depth ')
    plotter.plotLearningCurve(clf_tree, title=title + 'Max Depth', xTrain=xTrain, yTrain=yTrain) 

 
    clf_tree = DecisionTreeClassifier(random_state=util.randState) 
    clf_tree.set_params(**params)
    clf_tree.max_depth = 8
    param_range = [10, 50, 75, 100]
    param = 'min_samples_leaf'

    plotter.plotValidationCurve(clf_tree, xTrain, yTrain, param, param_range, graphTitle=title + ' Min Samples Leaf ')
  
    clf_tree.min_samples_leaf = 10 
    title = 'Heart'
    # plotter.plotLearningCurve(clf_tree, title=title + 'Min Samples Leaf', xTrain=xTrain, yTrain=yTrain) 
    plotter.plotLearningCurve(clf_tree, title=title, xTrain=xTrain, yTrain=yTrain) 
    clf_tree.fit(xTrain, yTrain)
    plotter.plotConfusion(clf_tree, title, ['Diameter narrowing ', 'Diameter not narrowing'], xTest, yTest)
 
def adult(dataType):
    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
    title =  '{0} Decision Tree'.format(dataType)
    xLabel = 'Depth'
    scoreList = util.ScoreList(xLabel)
 
   
    param_range = list(range(1, 20))
    param = 'max_depth'
    params = {'class_weight': None, 'criterion': 'entropy', 'max_features': None, 'min_samples_leaf': 50, 'splitter': 'best'}
  
    clf_tree = DecisionTreeClassifier(random_state=util.randState) 
    clf_tree.set_params(**params)    
    plotter.plotValidationCurve(clf_tree, xTrain, yTrain, param, param_range, graphTitle=title + ' Max Depth ')
    plotter.plotLearningCurve(clf_tree, title=title + 'Max Depth', xTrain=xTrain, yTrain=yTrain) 

 
    clf_tree = DecisionTreeClassifier(random_state=util.randState) 
    clf_tree.set_params(**params)
    clf_tree.max_depth = 5
    param_range = [10, 50, 75, 100]
    param = 'min_samples_leaf'
    plotter.plotValidationCurve(clf_tree, xTrain, yTrain, param, param_range, graphTitle=title + ' Min Samples Leaf ')
  
    clf_tree.min_samples_leaf = 50
    title = 'Adult'
    # plotter.plotLearningCurve(clf_tree, title=title + 'Min Samples Leaf', xTrain=xTrain, yTrain=yTrain) 
    plotter.plotLearningCurve(clf_tree, title=title, xTrain=xTrain, yTrain=yTrain) 
    clf_tree.fit(xTrain, yTrain)
    plotter.plotConfusion(clf_tree, title, ['>50K', '<=50K'], xTest, yTest)
 


def run(dataType):
    
    if dataType == 'Heart':
        heart(dataType)
    else:
        adult(dataType)

    return 