# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:21:05 2017

@author: arden
"""

import numpy as np
from sklearn.svm import SVC, OneVsRestClassifier, LabelBinarizer

x = np.array([[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]])
y = [0,0,1,1,2]
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
print(classif.fit(x,y).predict(x))

y = LabelBinarizer().fit_transform(y)
print(classif.fit(x, y).predict(x))

from sklearn.preprocessing import MultiLabelBinarizer
y = MultiLabelBinarizer().fit_transform(x)
print(classif.fit(x,y).predict(x))
