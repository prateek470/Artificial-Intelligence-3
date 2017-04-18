# @author: Prateek Gupta, Aman Mishra

import pandas as pd 
import numpy as np
import math
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

def data_input(filename):
#     Read data to a dataframe
    data = pd.read_csv("forestfires.csv")
    return data

# Preprocessing data includes
# 1. 1-to-C encoding as described in the paper
# Change output burned area values to log
def preprocessing(data):
#     For 1-to-C encoding
    data = pd.get_dummies(data) 
#     Change area data to log form
    data['area'] = np.log(data['area']+1)
    return data

# Select feature based on feature name
def feature_selection(data,feature):
    if feature=='STFWI':
        X = data.iloc[:,list(range(6))+list(range(10,29))].values
    elif feature=='STM':
        X = data.iloc[:,list(range(2))+list(range(6,29))].values
    elif feature=='FWI':
        X = data.iloc[:,2:6].values
    elif feature=='M':
        X = data.iloc[:,6:10].values
    else:
        X = data.iloc[:,6:10].values
    return X

# As per paper, feature vector should be standardized
def standardizedX(X):
    scaler = StandardScaler().fit(X)
    standardizedX = scaler.transform(X)
    return standardizedX

# Tune hyperparameter gamma and choose best gamma for model training
def hyperparameter_tuning(X, y):
	# Choose value of hyper parameter from below values of gamma
    gammas = [2**-1, 2**-3, 2**-5, 2**-7, 2**-9]
    classifier = GridSearchCV(estimator=svm.SVR(), cv=10, param_grid=dict(gamma=gammas))

    kf = KFold(n_splits=10, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
    return classifier

# 10- fold cross validation and error evaluation
# Loop of 30 to see 300 validations with shuffle on and off
# Loop of 10 to check the confidence interval - paper doesn't describe much on the 
# confidence interval. This works only for shuffle On.
def cross_validation_evaluation(X, y):
    mean_error, mad_error = 0, 0
    count = 0
    mean_min, mad_min = 100, 100
    mean_max, mad_max = 0, 0
    
    classifier = hyperparameter_tuning(X, y)
    model=svm.SVR(kernel='rbf', gamma=classifier.best_estimator_.gamma)
    
    for j in range(5):
        for i in range(30):
            kf = KFold(n_splits=10, random_state=None, shuffle=True)
            for train_index, test_index in kf.split(X):
                count += 1
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train,y_train)
                mean_error += mean_squared_error(np.exp(y_test)-1, np.exp(model.predict(X_test))-1)
                mad_error += mean_absolute_error(np.exp(y_test)-1,np.exp(model.predict(X_test))-1)
        mean_min = min(mean_min, (mean_error/count)**0.5)
        mean_max = max(mean_max, (mean_error/count)**0.5)
        mad_min = min(mad_min, (mad_error/count))
        mad_max = max(mad_max, (mad_error/count))
    RMSE = (mean_error/count)**0.5
    MAD = mad_error/count
    return RMSE, MAD, mean_min, mean_max, mad_min, mad_max
    
if __name__ == '__main__':
    data = data_input('forestfires.csv')
    data = preprocessing(data)
    features = ['STFWI','STM','FWI','M']
    y = data['area'].values
    data.__delitem__('area')
    for feature in features:
        X = feature_selection(data,feature)
        X = standardizedX(X)
        RMSE, MAD, mean_min, mean_max, mad_min, mad_max = cross_validation_evaluation(X, y)
        print "For feature: "+feature
        print 'RMSE +- Confidence Interval: '+"{0:.2f}".format(RMSE)+' +- '+"{0:.2f}".format(max(RMSE-mean_min, mean_max-RMSE))
        print 'MAD +- Confidence Interval: '+"{0:.2f}".format(MAD)+' +- '+"{0:.2f}".format(max(MAD-mad_min, mad_max-MAD))