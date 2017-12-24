# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:16:48 2017

@author: arden
"""

from sklearn import svm, datasets
import pickle
from sklearn.externals import joblib

clf = svm.SVC()
iris = datasets.load_iris()
clf.fit(iris.data, iris.target)

#using pickling strategy
s = pickle.dumps(clf)
clf2 = pickle.load(s)

#to confirm
print("Pickle method works:", (clf == clf2))

# using internal sklearn example
joblib.dump(clf, 'examplefile.pkl')
clf3 = joblib.load('examplefile.pkl')

#to confirm
print("Sklearn internal method works:" (clf == clf3))


#TO NOTE for future use joblib internally because pickle has security issues
