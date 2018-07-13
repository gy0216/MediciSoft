from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# load the iris datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target
sample = [[6, 4, 6, 2],]
# fit a Naive Bayes model to the data
naive = GaussianNB()
naive.fit(X,y)
print(naive)
# make predictions
predicted_value = naive.predict(sample)
print()
print("--------------------------------predict---------------------------------")
print(iris.target_names[predicted_value])
print(naive.predict_proba(sample))
