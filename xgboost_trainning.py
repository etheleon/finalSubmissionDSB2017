import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from sklearn import model_selection
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split

df = pd.read_csv("withlabels.csv")
df = df.dropna()
# df_test = df[pd.isnull(df).any(axis=1)]
# print(len(df_test))

labels = df['cancer']
X = df.drop(labels='cancer', axis=1)
#
# X_std = StandardScaler().fit_transform(X)
# x_pca = PCA(n_components=45).fit_transform(X_std)


# # We have not tried the TSNE one yet
# # x_tsne = TSNE(learning_rate=500).fit_transform(x)
#
# y = labels.as_matrix()
# #
# # # Stratified Cross-Validation
# skf = StratifiedKFold(n_splits=5, random_state=2048, shuffle=True)
#
# result = []
# clfs = []


# for train_index, test_index in skf.split(x_pca, y):
#     trn_x, val_x = x_pca[train_index, :], x_pca[test_index, :]
#     trn_y, val_y = y[train_index], y[test_index]
#
#     # XGBoost Model clf = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=96,
#     # learning_rate=0.03756, nthread=8, subsample=0.85, colsample_bytree=0.9, seed=96)
#     clf = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=96, learning_rate=0.03757, nthread=8,
#                            subsample=0.85, colsample_bytree=0.9, seed=96)
#     # max_depth=5 Public score = ?
#     # max_depth=4 Public score = 0.54721
#     # max_depth=3 Public score = 0.55193
#
#     # Training Model
#     clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=100)
#     result.append(clf.best_score)
#     clfs.append(clf)
#
# pickle.dump(clf, open( "model.pkl", "wb" ))
# print(result)


X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.33)
xgb_model = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=96, learning_rate=0.03757, nthread=8,
                           subsample=0.85, colsample_bytree=0.9, seed=96)
xgb_model.fit(X_train,y_train)

y_val_pred = xgb_model.predict_proba(X_val)
log_loss(y_val, y_val_pred)

# ----------- For test set -------------
#
testfile = pd.read_csv("pcaed.csv")
test_pca = testfile[testfile['label'] == 2]
test_pca = test_pca.drop(labels='label', axis=1)
print(test_pca.head())

sample_submission = pd.read_csv('stage1_sample_submission.csv')
with open('model.pkl', 'rb') as pickle_file:
    model_xgb = pickle.load(pickle_file)
#
# df_test = df_test.drop(labels='cancer', axis=1)
# test_std = StandardScaler().fit_transform(df_test)
# test_pca = PCA(n_components=45).fit_transform(test_std)
#
# test = np.array(test_pca)

pred = model_xgb.predict(test_pca)
sample_submission['cancer'] = pred

sample_submission.to_csv("submit1.csv", index=False)

print(sample_submission.head())

