import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib 
import pickle
import warnings
df=pd.read_csv("C:\\Users\\shaik\\OneDrive\\Desktop\\major\\crop_app\\Crop_recommendation.csv")
print(df.head())
if df['N'].all()>90:
    print(df['N'])
df.isnull().sum()
x = df.drop('label', axis = 1)
y = df['label']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, stratify = y, random_state = 1)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
logistic_acc = accuracy_score(y_test, y_pred)
print("Accuracy of logistic regression is " + str(logistic_acc))
from sklearn.tree import DecisionTreeClassifier
model_2 = DecisionTreeClassifier(criterion='entropy',max_depth = 6, random_state = 2)
model_2.fit(x_train, y_train)
y_pred_2 = model_2.predict(x_test)
decision_acc = accuracy_score(y_test, y_pred_2)
print("Accuracy of decision  tree is " + str(decision_acc))
from sklearn.naive_bayes import GaussianNB
model_3 = GaussianNB()
model_3.fit(x_train, y_train)
y_pred_3 = model_3.predict(x_test)
naive_bayes_acc = accuracy_score(y_test, y_pred_3)
print("Accuracy of naive_bayes is " + str(naive_bayes_acc))
from sklearn.ensemble import RandomForestClassifier
model_4 = RandomForestClassifier(n_estimators = 25, random_state=2)
model_4.fit(x_train.values, y_train.values)
y_pred_4 = model_4.predict(x_test)
random_fore_acc = accuracy_score(y_test, y_pred_4)
print("Accuracy of Random Forest is " + str(random_fore_acc))
file_name = "C:\\Users\\shaik\\OneDrive\\Desktop\\major\\crop_app\\crop_app"
with open(file_name,"wb") as f:
    joblib.dump(model_4,f)
app = joblib.load(file_name)
arr = [[90,42,43,20.879744,82.002744,6.502985,202.935536]]
acc = app.predict(arr)
print(acc)
Pkl_Filename = "C:\\Users\\shaik\\OneDrive\\Desktop\\major\\crop_app\\Pickle.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model_4, file)
with open(Pkl_Filename, 'rb') as file:  
    Pickled_Model = pickle.load(file)
Pickled_Model
RandomForestClassifier
RandomForestClassifier(n_estimators=25, random_state=2)