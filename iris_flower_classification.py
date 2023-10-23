import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# data preparation

df=pd.read_csv('Iris.csv')
df.drop('Id',axis=1,inplace=True)
sp={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
df.Species=[sp[i] for i in df.Species]
print(df)

X=df.iloc[:,0:4]
print(X)

y=df.iloc[:,4]
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

#traning model

knn = KNeighborsClassifier(n_neighbors = 3)
print(knn.fit(X_train, y_train))

#Evaluate model

y_pred = knn.predict(X_test)
print("y_pred")
print(y_pred)
print("y_test")
print(y_test)

#checkong accuracy

print("Accuracy",accuracy_score(y_test,y_pred))