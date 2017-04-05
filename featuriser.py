import os
import numpy as np
import dicom
import cv2
import mxnet as mx
import pandas as pd
from multiprocessing import Pool

img_rows = 512
img_cols = 512
smooth = 1.

files = [i + '.npy' for i in pd.read_csv('./missing.csv')['id']]


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
    np.save('./stage1.1_features/{}'.format(files[id_]), feats)
    return feats

if __name__ == '__main__':
    p = Pool(processes=15)
    p.map(calc_features, range(2, len(files)))
