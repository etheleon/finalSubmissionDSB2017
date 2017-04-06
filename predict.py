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

#Custom Modules
import BloodModel
import xgboost_model

img_rows = 512
img_cols = 512

smooth = 1.


def predict(sample,model,predictions):
    xgboost(sample,model)
    BloodPredict(sample,model)
    df.to_csv(predicitions, index=False)
    print(df.head())

if __name__ == '__main__':
    with open("./settings.json") as data_file:
      data = json.load(data_file)
    sample = data["data"]["TRAIN_DATA_PATH"]["Samples"]
    predictions = data["data"]["predictions"]
    model = data["data"]["model"]
    predict(sample,model,predictions)
