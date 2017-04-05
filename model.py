import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
bf_df = pd.read_csv('./bloodData.csv')
X_train, X_test, y_train, y_test = train_test_split(bf_df[['blood']], 
                                                    bf_df.cancer, 
                                                    random_state = 12345,
                                                   train_size = 0.8,
                                                   stratify = bf_df.cancer)
print('Training patients:{}, testing patients:{}'.format(X_train.shape[0], X_test.shape[0]))

#clf = SVC(probability=True)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
X_pred = clf.predict_proba(X_test)
X_pred = np.array([i[1] for i in X_pred])
test = log_loss(y_test, X_pred)
print(test)

df = pd.read_csv('./bloodSubmit.csv')
df2 = pd.read_csv('./stage1_sample_submission.csv')
dfa = []
for x in df2['id']:
	for i,y in enumerate(df['id']):
		if x == y:
			print(y)
			dfa.append(df['blood'][i])
			break

dfa = np.array(dfa)
pred = clf.predict_proba(dfa.reshape(-1, 1))
df2['cancer'] = pred[:,1:]
df2.to_csv('./subm2.csv', index=False)
