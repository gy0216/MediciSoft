# Support Vector Machine
from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC
# load the iris datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target
# fit a SVM model to the data
svm = SVC()
svm.fit(X,y)
print()
print("--------------------model----------------------")
print(svm)
# make predictions
print()
print("----------------------predict-------------------")
sample = [[6, 4, 6, 2],]
predicted_value = svm.predict(sample)
print(iris.target_names[predicted_value])
