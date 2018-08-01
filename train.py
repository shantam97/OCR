import numpy as np
import cv2 as cv
from sklearn import svm
from sklearn import tree
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import cPickle as cp
from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier


img = cv.imread("digits.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]


x = []
for i in range(50):
    for j in range(100):
        fd = hog(cells[i][j], orientations=9, pixels_per_cell=(
            10, 10), cells_per_block=(1, 1), visualise=False)
        x.append(fd)
c = np.array(x, 'float64')

labels = []
for i in range(10):
    labels = labels + [i] * 500


clf = KNeighborsClassifier()
x_train, x_test, y_train, y_test = tts(c, labels, test_size=0.01)
clf.fit(x_train, y_train)


print accuracy_score(y_test, clf.predict(x_test))


with open("knn.mdl", 'wb') as fp:
    cp.dump(clf, fp)
