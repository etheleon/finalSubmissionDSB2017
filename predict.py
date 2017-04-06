#In_Built Library
import os
import glob
import json

#Data Analysis 
import pandas as pd

#Custom Modules
import BloodModel
import xgboost_model

def predict(sample,model,predictions):
    pred1 = xgboost(sample,model)
    pred2 = BloodPredict(sample,model)
    for i,x in pred1:
        if x>0.3 and pred2[i]>0.3:
            pred1[i] = 0.9999999999
    df.to_csv(pred1, index=False)
    print(df.head())
    
if __name__ == '__main__':
    with open("./settings.json") as data_file:
      data = json.load(data_file)
    sample = data["data"]["TRAIN_DATA_PATH"]["Samples"]
    predictions = data["data"]["predictions"]
    model = data["data"]["model"]
    predict(sample,model,predictions)
