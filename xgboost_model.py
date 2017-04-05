img_rows = 512
img_cols = 512

smooth = 1.

import os
import pickle
import numpy as np
import dicom
import glob
from matplotlib import pyplot as plt
import cv2
import mxnet as mx
import pandas as pd
from sklearn import cross_validation
import xgboost as xgb

def train_xgboost():
    df = pd.read_csv('./stage1_labels.csv')
    print(df.head())
    
    x = np.array([np.mean(np.load('stage1.1_features/%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
    print(x.shape)
    y = df['cancer'].as_matrix()

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
    
                                                                test_size=0.20)
    clf = xgb.XGBRegressor(max_depth=10,
                           n_estimators=1500,
                           min_child_weight=9,
                           learning_rate=0.005,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)
    # clf = xgb.XGBRegressor(max_depth=10,
    #                        n_estimators=1500,
    #                        min_child_weight=9,
    #                        learning_rate=0.05,
    #                        nthread=8,
    #                        subsample=0.80,
    #                        colsample_bytree=0.80,
    #                        seed=4242)

    # clf = xgb.XGBRegressor(max_depth=20,
    #                        n_estimators=10000,
    #                        min_child_weight=20,
    #                        learning_rate=0.05,
    #                        nthread=8,
    #                        subsample=0.80,
    #                        colsample_bytree=0.80,
    #                        seed=4242)

    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
    pickle.dump(clf, open("xgboostmodel.dat", "wb"))
    return clf

if __name__ == '__main__':
  train_xgboost()
