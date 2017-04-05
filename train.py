#!/usr/bin/env python

#Custom Modules 
from featuriser import calc_features
from preprocessing import mask_generator
from xgboost_model import train_xgboost

#logging.basicConfig(level=logging.INFO)
logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)

def features(patientFolder):
    finial_images = mask_generator(patientFolder)
    return calc_features(finial_images)
    
if __name__ == '__main__':
    with open("./settings.json") as data_file:
        data = json.load(data_file)
    INPUT_FOLDER = data["data"]["raw"]["DSB"]
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    #patients = ['00cba091fa4ad62cc3200a657aeb957e']
    patientFolders = ["{}/{}".format(INPUT_FOLDER, patient) for patient in patients]
    p = mp.Pool(processes = 23)
    p.map(features, patientFolders)
    train_xgboost()
    #preprocessing2(patientFolders[0])
