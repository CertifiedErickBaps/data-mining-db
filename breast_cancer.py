# Python ≥3.5 is required
import sys

assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"
import os

import matplotlib as mpl

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parameters = {'criterion': ['entropy', 'gini'],
              'min_samples_split': [2, 3, 4],
              'max_depth': range(1, 20, 2),
              'max_leaf_nodes': list(range(2, 100))
              }
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), parameters, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)
print(grid_search_cv.best_estimator_)

tree1 = grid_search_cv.best_estimator_

y_pred = tree1.predict(X_test)
result = accuracy_score(y_test, y_pred)

print("Accuracy is", result * 100)

from graphviz import Source
from sklearn.tree import export_graphviz

export_graphviz(
    tree1,
    out_file=os.path.join(IMAGES_PATH, "cancer_tree.dot"),
    feature_names=cancer.feature_names,
    class_names=cancer.target_names,
    rounded=True,
    filled=True
)
Source.from_file(os.path.join(IMAGES_PATH, "cancer_tree.dot"))
