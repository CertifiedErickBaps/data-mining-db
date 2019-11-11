# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
import os

import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)


from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

cancer=load_breast_cancer()
X = cancer.data
y = cancer.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree1 = DecisionTreeClassifier(max_depth=4, criterion='entropy')
tree1.fit(X_train, y_train)

y_pred = tree1.predict(X_test)
result = accuracy_score(y_test, y_pred)

print("Accuracy is", result*100)

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

