
import util
import data 
import plotter
import searcher
import numpy as np
 
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def adult(dataType):
    title =  '{0} Ada Boost'.format(dataType)
    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
 
    param_range = list(range(1, 160, 10))
    param = 'n_estimators'
 

    # params = {'algorithm': 'SAMME.R'}
    clf = AdaBoostClassifier()
    # clf.set_params(**params)
    
    plotter.plotValidationCurve(clf, xTrain, yTrain, param, param_range, graphTitle=title) 
    clf.n_estimators = 40
    plotter.plotLearningCurve(clf, title=title, xTrain=xTrain, yTrain=yTrain) 
    title = 'Adult' 
    clf.fit(xTrain, yTrain)
    plotter.plotConfusion(clf, title, ['>50K', '<=50K'], xTest, yTest)

def heart(dataType):
    title =  '{0} Ada Boost'.format(dataType)
    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
 
    param_range = list(range(1, 160, 10))
    param = 'n_estimators'
 

    params = {'algorithm': 'SAMME.R'}
    clf = AdaBoostClassifier()
    clf.set_params(**params)
    
    plotter.plotValidationCurve(clf, xTrain, yTrain, param, param_range, graphTitle=title) 
    plotter.plotLearningCurve(clf, title=title, xTrain=xTrain, yTrain=yTrain) 
    title = 'Heart' 
    clf.fit(xTrain, yTrain)
    plotter.plotConfusion(clf, title, ['Diameter narrowing ', 'Diameter not narrowing'], xTest, yTest)
 
 

def run(dataType):
 
    if dataType == 'Heart':
        heart(dataType)
    else:
        adult(dataType)

    return
 