import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

credit_raw = pd.read_csv('creditcard_2023.csv')
credit_raw.shape
credit_raw.head()
credit_raw.describe()
credit_raw.info()
credit_raw['Class'].unique()
credit_raw.isnull().sum()
credit_raw.corr()
credit = credit_raw.drop(['id'], axis=1)


X = credit.drop(['Class'], axis=1)
y = credit['Class']
from sklearn.model_selection import train_test_split
X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=0.4, random_state=21)
X_valid, X_test, y_valid, y_test = train_test_split(X_test_valid, y_test_valid, test_size=0.5, random_state=21)

#
# perfrom xgboost
#
from sklearn.metrics import accuracy_score
import xgboost as xgb

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

params = {
    "gamma": Real(0, 0.5, 'uniform'),
    "learning_rate": Real(0.03, 0.3, 'uniform'),    # default 0.1
    "max_depth": Integer(2, 6),                     # default 3
    "n_estimators": Integer(100, 150),              # default 100
}
clf_xgb_search = xgb.XGBClassifier()
search = BayesSearchCV(clf_xgb_search, search_spaces=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=1,
                            return_train_score=True)
search.fit(X_train, y_train)
search.best_estimator_
search.best_params_

clf_xgb = xgb.XGBClassifier(params=search.best_params_)
start = time.time()
clf_xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
end = time.time()
print(f'{int((end-start)/3600):02}:{int((end - start)/60) % 60:02}:{(end - start) % 60:02}')

y_pred_valid = clf_xgb.predict(X_valid)
y_pred = clf_xgb.predict(X_test)
xgb_train = round(clf_xgb.score(X_train, y_train) * 100, 2)
xgb_valdation = round(accuracy_score(y_pred_valid, y_valid) * 100, 2)
xgb_accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
print("Training Accuracy    :", xgb_train)
print("Validation    :", xgb_valdation)
print("Model Accuracy Score :", xgb_accuracy)
from sklearn.metrics import confusion_matrix
cm_xgb = confusion_matrix(y_test, y_pred)
cm_xgb
#[0,0] is true negative, [0,1] is false positive, [1,0] is false negative, [1,1] is true positive
print('Accuracy = ' + str((cm_xgb[0, 0]+cm_xgb[1, 1])/(cm_xgb[0, 0]+cm_xgb[0, 1]+cm_xgb[1, 0]+cm_xgb[1, 1])))
print('Recall = ' + str((cm_xgb[1, 1])/(cm_xgb[1, 0]+cm_xgb[1, 1])))
print('Precision = ' + str(cm_xgb[0, 0]/(cm_xgb[0, 0]+cm_xgb[0, 1])))
print('False positive rate = ' + str((cm_xgb[0, 1])/(cm_xgb[1, 0]+cm_xgb[0, 1])))
#
#
#
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
X.shape
clf_nn = Sequential([
    Dense(units=64, input_shape=(29,), activation='relu'),
    Dense(units=8, activation='relu'),
    Dense(units=1, activation='sigmoid')
])
clf_nn.summary()
clf_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start = time.time()
clf_nn.fit(x=X_train, y=y_train, batch_size=128, epochs=16)
end = time.time()
print(f'{int((end-start)/3600):02}:{int((end - start)/60) % 60:02}:{(end - start) % 60:02}')

_, nn_train = clf_nn.evaluate(X_train, y_train, verbose=0)
nn_train = round(nn_train * 100, 2)
_, nn_accuracy = clf_nn.evaluate(X_valid, y_valid, verbose=0)
nn_accuracy = round(nn_accuracy * 100, 2)
print("Training Accuracy    :",nn_train)
print("Model Accuracy Score :",nn_accuracy)
y_pred = clf_nn.predict(X_test, verbose=0).reshape((-1,)).round()
cm_nn = confusion_matrix(y_valid, y_pred)
cm_nn
print('Accuracy = ' + str((cm_nn[0, 0]+cm_nn[1, 1])/(cm_nn[0, 0]+cm_nn[0, 1]+cm_nn[1, 0]+cm_nn[1, 1])))
print('Recall = ' + str((cm_nn[1, 1])/(cm_nn[1, 0]+cm_nn[1, 1])))
print('Precision = ' + str(cm_nn[0, 0]/(cm_nn[0, 0]+cm_nn[0, 1])))
print('False positive rate = ' + str((cm_nn[0, 1])/(cm_nn[1, 0]+cm_nn[0, 1])))