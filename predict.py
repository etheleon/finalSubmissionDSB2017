#In_Built Library
import os
import glob
import json

#Data Analysis 
import numpy as np
import pandas as pd

#Custom Modules
import BloodModel
import xgboost_model

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
