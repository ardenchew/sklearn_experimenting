# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import numpy as np
from sklearn import datasets, svm
#from sklearn import svm

iris = datasets.load_iris()
digits = datasets.load_digits()

#black box estimator for predicitive science
#gamma chosen ARBITRATILY, use grid search or cross validation to choose by learning

clf = svm.SVC(gamma=0.001, C=100.)

#machine learning algorithms all require a training set
#passing estimator through fitting method
clf.fit(digits.data[:-1], digits.target[:-1])


