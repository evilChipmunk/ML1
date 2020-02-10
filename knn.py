import util 
import searcher
import data
import plotter
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
 
def heart(dataType):

    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
    xLabel = 'K'
    scoreList = util.ScoreList(xLabel)
    title =  '{0} KNN'.format(dataType)
    
    # searcher.searchKNN(xTrain, yTrain, xTest, yTest)
    params = {'algorithm': 'auto', 'p': 1, 'weights': 'uniform'}
    params = {'algorithm': 'ball_tree', 'p': 1, 'weights': 'distance'} 
    # params = searcher.searchKNN(xTrain, yTrain, xTest, yTest)

    param = 'n_neighbors'
    param_range = list(range(1, 50)) #np.linspace(1, 50, 50) 
  
    clf = KNeighborsClassifier()
    clf.set_params(**params)

    plotter.plotValidationCurve(clf, xTrain, yTrain, param, param_range, graphTitle=title)

    clf = KNeighborsClassifier()
    clf.set_params(**params)
    clf.n_neighbors = 12
    plotter.plotLearningCurve(clf, title=title, xTrain=xTrain, yTrain=yTrain)
    # plotter.plotAll(clf, title, param, param_range, xTrain, yTrain, xTest, yTest)
    title = 'Heart' 
    clf.fit(xTrain, yTrain)
    plotter.plotConfusion(clf, title, ['Diameter narrowing ', 'Diameter not narrowing'], xTest, yTest)
 
      
    
def adult(dataType):

    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
    xLabel = 'K'
    scoreList = util.ScoreList(xLabel)
    title =  '{0} KNN'.format(dataType)
    
    # searcher.searchKNN(xTrain, yTrain, xTest, yTest)
    params = {'algorithm': 'auto', 'p': 1, 'weights': 'uniform'}
    params = {'algorithm': 'ball_tree', 'p': 1, 'weights': 'distance'}
    params = {'algorithm': 'auto', 'p': 1, 'weights': 'distance'} 
    # params = searcher.searchKNN(xTrain, yTrain, xTest, yTest)

    clf = KNeighborsClassifier()
    clf.set_params(**params)
    param = 'n_neighbors'
    param_range = list(range(1, 50)) #np.linspace(1, 50, 50) 
    plotter.plotValidationCurve(clf, xTrain, yTrain, param, param_range, graphTitle=title)
  
    clf = KNeighborsClassifier()
    clf.set_params(**params)
    clf.n_neighbors = 5 
    plotter.plotLearningCurve(clf, title=title, xTrain=xTrain, yTrain=yTrain)
    # plotter.plotAll(clf, title, param, param_range, xTrain, yTrain, xTest, yTest)

    title = 'Adult' 
    clf.fit(xTrain, yTrain)
    plotter.plotConfusion(clf, title, ['>50K', '<=50K'], xTest, yTest)

def run(dataType):

    if dataType == 'Heart':
        return
        heart(dataType)
    else:
        adult(dataType)
 