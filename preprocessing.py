#!/usr/bin/env python
import os
import logging
import multiprocessing as mp
import time
#import argparse

import json

#Image processing
import dicom
from skimage import measure, morphology

#Data Stuff
import numpy as np
from sklearn.cluster import KMeans

def load_scan(path):
    '''
    sorts the slices, via ImagePositionPatient z-axis 3rd element, calculates thickness
    '''
    logger.info("loading dicoms from {}".format(path))
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(slices):
    '''
    get the pixel's HU units
    '''
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def mask_generator(patientFolder):
    '''
    Takes dicom slices and generates a numpy array
    Taken from https://www.kaggle.com/c/data-science-bowl-2017/details/tutorial#u-net-segmentation-approach-to-cancer-diagnosis
    '''
    def erosionDilation(thresh_img):
        logger.debug("Labeling")
        eroded     = morphology.erosion(thresh_img,np.ones([4,4]))
        dilation   = morphology.dilation(eroded,np.ones([10,10]))
        labels     = measure.label(dilation)
        logger.debug("Finished labeling")
        return labels

    def contrastup(img):
        middle = img[100:400,100:400]
        mean = np.mean(middle)
        max = np.max(img)
        min = np.min(img)
        #move the underflow bins
        img[img==max]=mean
        img[img==min]=mean

        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img<threshold,1.0,0.0)
        return thresh_img

    def roi(img, labels):
        regions = measure.regionprops(labels)
        if(len(regions) >= 2): #with at least 2 regions ie. each lung
            good_labels = []
            for prop in regions:
                B = prop.bbox
                if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                    good_labels.append(prop.label)
            mask = np.ndarray([512,512],dtype=np.int8)
            mask[:] = 0
            #
            #  The mask here is the mask for the lungs--not the nodes
            #  After just the lungs are left, we do another large dilation
            #  in order to fill in and out the lung mask
            #
            for N in good_labels:
                mask = mask + np.where(labels==N,1,0)
            mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
            if np.amax(mask) == 0 & np.amin(mask)== 0: #cause some of the slices are actually the abdomen
                img = img * 0
                return img
            else:
                img = mask*img
                new_mean = np.mean(img[mask>0])
                new_std = np.std(img[mask>0])

                old_min = -100     # background color
                img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
                img = img-new_mean
                img = img/new_std
                return img
        else:
            img = img*0
            return img

    start_time = time.time()
    first_patient = load_scan(patientFolder)
    final_images = np.ndarray([int(len(first_patient)/3),3,512,512],dtype=np.float32)

    fullRange = [i for i in range(0,len(first_patient),3)]
    fullRange.pop() #throw away the tail might not have enough to build RGB
    for i in fullRange:
        logger.info("Processing {} {}th slice".format(patientFolder, i))
        try:
            imgs = [get_pixels_hu(first_patient)[i], get_pixels_hu(first_patient)[i+1],get_pixels_hu(first_patient)[i+2]]
            for x,img in enumerate(imgs):
                thresh_img = contrastup(img)
                labels     = erosionDilation(thresh_img)
                masked     = roi(img, labels)
                final_images[int(i/3),x] = masked
        except:
            logger.exception("Bummer I have exception")
    outputFile = "{}.npy".format(patientFolder)
    logger.info("saving to {}".format(outputFile))
    np.save(outputFile,final_images)
    print("Preprocessing is finished in --- %s seconds ---" % (time.time() - start_time))
    return final_images
