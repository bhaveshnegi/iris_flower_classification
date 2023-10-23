from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# data preparation
iris = load_iris()
print("Target")
print(iris.target)
print("Target Names")
print(iris.target_names)
print("Feature Names")
print(iris.feature_names)
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4)

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