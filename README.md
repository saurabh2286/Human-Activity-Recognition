# Human-Activity-Recognition
import numpy as np
import pandas as pd
file1= r"C:\Users\saura\Desktop\Python Fundamentals\Data\HAR-train.csv"
file2= r"C:\Users\saura\Desktop\Python Fundamentals\Data\HAR-test.csv"
h_train=pd.read_csv(file1)
h_test = pd.read_csv(file2)
h_train['data']='train'
h_test['data']='test'
#Merging train and test dataset for data prepartion
h_all=pd.concat([h_train,h_test],axis=0)
h_all.info()
h_all.isnull().sum()
h_all['Activity'].value_counts()
h_all.select_dtypes(['object']).columns
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.countplot('Activity',data=h_all)
h_train=h_all.loc[h_all['data']=='train']
h_test=h_all.loc[h_all['data']=='test']
h_train = h_train.drop('data',axis=1)
h_test = h_test.drop('data',axis=1)
del h_all
X_train=h_train.drop('Activity',axis=1)
X_test= h_test.drop('Activity',axis=1)
y_train=h_train['Activity']
y_test=h_test['Activity']
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
print(confusion_matrix(y_test,rf_pred))
print(classification_report(y_test,rf_pred))
print(accuracy_score(y_test, rf_pred))
