#In_Built Library
import os
import glob
import json

#Data Analysis 
import pandas as pd

#Custom Modules
from BloodModel import bloodPredict
from xgboost_model import xgboostModel

def predict(sample,model,predictions):
    pred1 = xgboostModel(sample,model)
    # pred2 = BloodPredict(sample,model)
    # for i,x in pred1:
    #     if x>0.3 and pred2[i]>0.3:
    #         pred1[i] = 0.9999999999
    pred1.to_csv(predictions, index=False)
    print(pred1.head())
    
if __name__ == '__main__':
    with open("./settings.json") as data_file:
      data = json.load(data_file)
    sample = data["TRAIN_DATA_PATH"]["Samples"]
    predictions = data["predictions"]
    model = data["model"]
    predict(sample,model,predictions)
