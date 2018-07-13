
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# load iris the datasets
iris = datasets.load_iris()
# fit a k-nearest neighbor model to the data
X,y = iris.data, iris.target
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
print()
print("--------------------------model-------------------------------")
print(knn)
# make predictions
sample = [[6,4,6,2],]
print()
print("-------------------------result-------------------------------")
predicted_value = knn.predict(sample)
print(iris.target_names[predicted_value])
print(knn.predict_proba(sample))
print()
print("--------------------chagne weight----------------------------")
knn = KNeighborsClassifier(n_neighbors=100, weights='distance')
knn.fit(X, y)
predicted_value = knn.predict(sample)
print(iris.target_names[predicted_value])
print(knn.predict_proba(sample))
