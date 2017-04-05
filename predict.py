#In_Built Library
import os
import pickle
import glob

import json

#Data Analysis 
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#Image Analysis
import dicom
import cv2

#Machine Learning
import mxnet as mx
from sklearn import cross_validation
import xgboost as xgb

img_rows = 512
img_cols = 512

smooth = 1.

def predict(sample,model,predictions):
    try:
        clf = pickle.load(open(model, "rb"))
        print("Opening Model")
    except:
        clf = train_xgboost()

    df = pd.read_csv(sample)
    pred = []
    for id_ in df['id'].tolist():
            try:
                x = np.mean(np.load(sample+'%s.npy' % str(id_)), axis=0)
                pred.append(clf.predict(np.array([x]))[0])
            except FileNotFoundError:
                pred.append(0.25912670007158195)

    df['cancer'] = pred
    df.to_csv(predicitions, index=False)
    print(df.head())

if __name__ == '__main__':
    with open("./settings.json") as data_file:
      data = json.load(data_file)
    sample = data["data"]["TRAIN_DATA_PATH"]["Samples"]
    predictions = data["data"]["predictions"]
    model = data["data"]["model"]
    predict(sample,model,predictions)
