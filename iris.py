from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import mean_absolute_error


iris_data, iris_label = load_iris(return_X_y=True)
dtc = DecisionTreeClassifier()
dtc.fit(iris_data, iris_label)
preds = dtc.predict(iris_data)
export_graphviz(dtc, out_file="iris_tree")

print("Done!")