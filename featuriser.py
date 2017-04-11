#!/usr/bin/env python

#In_Built Libaries
import glob
import multiprocessing as mp

#Data Analysis
import numpy as np
import pandas as pd

#Machine Learning
import mxnet as mx

img_rows = 512
img_cols = 512
smooth = 1.

def get_extractor():
    model = mx.model.FeedForward.load('/w/data/one/stage1/resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol,
    numpy_batch_size=64, arg_params=model.arg_params, aux_params=model.aux_params,
    allow_extra_params=True)
    return feature_extractor

def readFiles(file):
    try:
        print("Reading {}".format(file))
        image = np.load(file)
        summed = np.mean(image, axis=0)
        return (file, summed)
    except:
        print("This file is problematic: {}".format(file))

if __name__ == '__main__':
    net = get_extractor()
    files = glob.glob("/w/data/two/stage2/data/*.npy")
    results= mp.Pool(20).map(readFiles, files)
    names = [i[0] for i in results]
    batchArrays = [i[1] for i in results]
    fullstack = np.stack(batchArrays, axis=0)
    feats = net.predict(fullstack)
    df = pd.DataFrame(np.stack(feats, axis=0))
    df["filenames"] = names
    df.to_csv("/w/data/two/stage2/reprocessed_features.csv")
