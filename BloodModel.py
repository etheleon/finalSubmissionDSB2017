#Built-In Models
import pickle
import json 

#Data Analysis
import numpy as np
import pandas as pd

#Machine Learning
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss

def load_scan(data):
	blood = {"blood":[]}
	for path in data:
		blood_sum = []
		for slices in os.listdir(path):
			#print(path)
		    # Here are a list of simple features and ranges to look at in more detail
			feature_list = {
			    'blood': (30, 45),
			    'fat': (-100, -50),
			    'water': (-10, 10)
			}
			n_img = read_file(path + '/' + slices)
			blood_sum.append(calc_area(n_img, *feature_list['blood']))
		
		blood["blood"].append(sum(blood_sum))
	return pd.DataFrame(blood)

def calc_area(in_dcm, min_val, max_val):
    pix_area = np.prod(in_dcm.PixelSpacing)
    return pix_area*np.sum((in_dcm.pixel_array>=min_val) & (in_dcm.pixel_array<=max_val))

def bloodTrain(data,model):
	#Load the amount of blood in each slice
	bf_df = load_scan(data)
	
	#Split the data
	X_train, X_test, y_train, y_test = train_test_split(bf_df[['blood']], 
							    bf_df.cancer, 
							    random_state = 12345,
							   train_size = 0.8,
							   stratify = bf_df.cancer)
	
	print('Training patients:{}, testing patients:{}'.format(X_train.shape[0], X_test.shape[0]))
	
	#Models
	#clf = SVC(probability=True)
	clf = DecisionTreeClassifier()
	clf.fit(X_train, y_train)
	
	#Saving the model
	pickle.dumps(model,clf)
	
	#Model Evaluation
	X_pred = clf.predict_proba(X_test)
	X_pred = np.array([i[1] for i in X_pred])
	test = log_loss(y_test, X_pred)
	return test
	
def bloodPredict(data,model):
	#Load data
	X = load_scan(data)
	
	#Load Model
	clf = pickle.load(model)
	
	#Make Predicitons
	pred = clf.predict_proba(X)
	
	return pred[:,1:]
