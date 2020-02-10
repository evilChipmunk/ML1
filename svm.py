import util 
import searcher
import data
import plotter

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve, validation_curve
 
def createBaseSVC(): 
     return SVC(cache_size=5000, max_iter=5000, C = 1.0, gamma = 0.01, verbose=10)
     

def heart(dataType):
    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
       
    title =  '{0} SVM'.format(dataType)
  
    param_range = list(range(1,8))
    # polyparams = searcher.searchSVMPoly(xTrain, yTrain, xTest, yTest)
    # polyparams = {'kernel': 'poly', 'gamma': 'scale'}
    polyparams = {'C': 0.01, 'degree': 3, 'gamma': 10, 'kernel': 'poly'}
    clf = createBaseSVC()
    clf.set_params(**polyparams)
    plotter.plotValidationCurve(clf, xTrain, yTrain, 'degree', param_range, graphTitle=title + ' Poly Degree ')
    
    clf.degree = 3
    plotter.plotLearningCurve(clf, title=title + ' Poly degree ', xTrain=xTrain, yTrain=yTrain)
 
 
    # rbfParams = searcher.searchSVMRBF(xTrain, yTrain, xTest, yTest)   
    rbfParams = {'C': 1, 'gamma': 1, 'kernel': 'rbf'}
    clf = createBaseSVC()
    clf.set_params(**rbfParams)
    param = 'C' 
    param_range = [0.01,0.05,0.25, 0.5, 1]
    plotter.plotValidationCurve(clf, xTrain, yTrain, param, param_range, graphTitle=title + ' RBF - C ')
    clf.C = 0.5
    plotter.plotLearningCurve(clf, title=title + ' RBF', xTrain=xTrain, yTrain=yTrain)


    title = 'Heart' 
    clf.fit(xTrain, yTrain)
    plotter.plotConfusion(clf, title, ['Diameter narrowing ', 'Diameter not narrowing'], xTest, yTest)
    # param = 'kernel'
    # param_range = ['linear', 'rbf', 'sigmoid', 'poly']
    # # params = searcher.searchSVMLinear(xTrain, yTrain, xTest, yTest)
    # clf = SVC(cache_size=5000, degree=5, class_weight='balanced', max_iter=7000) 

    # plotter.plotValidationCurve(clf, xTrain, yTrain, param, param_range, graphTitle=title) 
    # clf.kernel = 'rbf'
    # plotter.plotLearningCurve(clf, title=title, xTrain=xTrain, yTrain=yTrain) 

def adult(dataType):
    package = data.createData(dataType)
    
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
    
    xLabel = 'Degrees' 
    title =  '{0} SVM'.format(dataType)
  
    param_range = list(range(1,8))
     
    # polyparams = searcher.searchSVMPoly(xTrain, yTrain, xTest, yTest) 
    polyparams = {'C': 0.1, 'degree': 1, 'gamma': 50, 'kernel': 'poly'} 
    clf = createBaseSVC()
    clf.set_params(**polyparams)
    plotter.plotValidationCurve(clf, xTrain, yTrain, 'degree', param_range, graphTitle=title + ' Poly degree ')
    clf.degree = 1
    plotter.plotLearningCurve(clf, title=title + ' Poly degree ', xTrain=xTrain, yTrain=yTrain)
 
 
    # rbfParams = searcher.searchSVMRBF(xTrain, yTrain, xTest, yTest) 
    rbfParams = {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'} 
 
    clf = createBaseSVC()
    clf.set_params(**rbfParams)
    param = 'C'
    # param_range = [0.01,0.05,1,10,50,100,200,300,500, 1000]
    param_range = [0.01,0.05,1,10,15]
    plotter.plotValidationCurve(clf, xTrain, yTrain, param, param_range, graphTitle=title+ ' RBF - C ')
    clf.C = 10
    plotter.plotLearningCurve(clf, title=title + ' RBF', xTrain=xTrain, yTrain=yTrain)
    
    title = 'Adult' 
    clf.fit(xTrain, yTrain)
    plotter.plotConfusion(clf, title, ['>50K', '<=50K'], xTest, yTest)
 

 



def run(dataType):     
    if dataType == 'Heart':
        heart(dataType)
    else:
        adult(dataType) 
   