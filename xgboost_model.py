#!/usr/bin/env python

#In-Built Files
import os
import pickle
import glob

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
from scipy.stats import gmean
import xgboost as xgb

img_rows = 512
img_cols = 512

smooth = 1.

def train_xgboost():
    df = pd.read_csv('./stage1_labels.csv')
    print(df.head())
    
    x = np.array([np.mean(np.load('stage1.1_features/%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
    print(x.shape)
    y = df['cancer'].as_matrix()
    
    skf = StratifiedKFold(n_splits=5, random_state=2048, shuffle=True)
    
    result =[]
    clfs = []
    for train_index, test_index in skf.split(x, y):
        trn_x, val_x = x[train_index,:], x[test_index,:]
        trn_y, val_y = y[train_index], y[test_index]

        #clf = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=96, learning_rate=0.03756, nthread=8, subsample=0.85, colsample_bytree=0.9, seed=96)
        clf = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=96, learning_rate=0.03757, nthread=8, subsample=0.85, colsample_bytree=0.9, seed=96)
        # max_depth=5 Public score = ?
        # max_depth=4 Public score = 0.54721
        # max_depth=3 Public score = 0.55193

        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=100)
        result.append(clf.best_score)
        clfs.append(clf)

    pickle.dump(clf, open("xgboostmodel.pkl", "wb"))
    return clfs, result

if __name__ == '__main__':
  train_xgboost()
