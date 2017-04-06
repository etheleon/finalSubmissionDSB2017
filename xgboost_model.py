#!/usr/bin/env python

#In-Built Files
import os
import pickle
import glob
import json

#Data Analysis Modules
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#Image Analysis
import dicom
import cv2

#Machine Learning Models
import mxnet as mx
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gmean
import xgboost as xgb

img_rows = 512
img_cols = 512

smooth = 1.

def train_xgboost(labels,features,model):
    df = pd.read_csv(labels)
    print(df.head())
    
    x = np.array([np.mean(np.load(features + '/%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
    
    #PCA and feature selection / We have not tried the TSNE one yet
    X_std = StandardScaler().fit_transform(x)
    x_pca = PCA(n_components = 13).fit(X_std)
    
    y = df['cancer'].as_matrix()
    
    #Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, random_state=2048, shuffle=True)
    
    result =[]
    clfs = []
    for train_index, test_index in skf.split(x, y):
        trn_x, val_x = x[train_index,:], x[test_index,:]
        trn_y, val_y = y[train_index], y[test_index]
        
        #XGBoost Model
        #clf = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=96, learning_rate=0.03756, nthread=8, subsample=0.85, colsample_bytree=0.9, seed=96)
        clf = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=96, learning_rate=0.03757, nthread=8, subsample=0.85, colsample_bytree=0.9, seed=96)
        # max_depth=5 Public score = ?
        # max_depth=4 Public score = 0.54721
        # max_depth=3 Public score = 0.55193
        
        #Training Model
        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=100)
        result.append(clf.best_score)
        clfs.append(clf)

    pickle.dump(clf, open(model, "wb"))
    return result

def xgboostModel(sample,model):
    #Opening up model
    try:
        clf = pickle.load(open(model, "rb"))
        print("Opening Model")
    except:
        clf = train_xgboost()
    
    #Make predicitions
    df = pd.read_csv(sample)
    pred = []
    for id_ in df['id'].tolist():
            try:
                x = np.mean(np.load(sample+'%s.npy' % str(id_)), axis=0)
                pred.append(clf.predict(np.array([x]))[0])
            except FileNotFoundError:
                #0.259126 is the mean in that population of test data that a person has cancer
                pred.append(0.25912670007158195)
    
    df['cancer'] = pred
    return pred

if __name__ == '__main__':
  with open("./settings.json") as data_file:
      data = json.load(data_file)
  labels = data["data"]["raw"]["Labels"]
  features = data["data"]["features"]
  train_xgboost(labels,features)
