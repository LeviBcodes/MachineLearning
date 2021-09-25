from sklearn.datasets import load_iris
iris = load_iris()
print(list(iris.target_names))

from sklearn import tree
classifier = tree.DesicionTreeClassifier()
classifier = classifier.fit(iris.data, iris.target)