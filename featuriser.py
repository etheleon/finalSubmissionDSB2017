#!/usr/bin/env python

#In_Built Libaries
import os
import glob
from multiprocessing import Pool

#Data Analysis
import numpy as np
import pandas as pd

#Image Processing
import dicom
import cv2

#Machine Learning
import mxnet as mx

img_rows = 512
img_cols = 512
smooth = 1.

def get_extractor():
    model = mx.model.FeedForward.load('./resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=64,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)

    return feature_extractor

net = get_extractor()

def calc_features(image):
    batch = np.mean(image, axis=0)
    batch = np.array([batch])
    print("{} is Found ".format(str(id_)))

    feats = net.predict(batch)
    print(feats.shape)
    np.save('./train_features/{}'.format(files[id_]), feats)
    return feats

if __name__ == '__main__':
    with open("./settings.json") as data_file:
      data = json.load(data_file)
    INPUT_FOLDER = data["data"]["raw"]["DSB"]
    p = Pool(processes=15)
    p.map(calc_features, glob.glob(INPUT_FOLDER))
