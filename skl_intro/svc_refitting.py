# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 22:57:41 2017

@author: arden
"""

import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)
x = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100);
x_test = rng.rand(5, 10)

cl = SVC()
clf.set_params(kernel='linear').fit(x, y)
clf.predict(x_test)

clf.set_params(kernel='rbf').fit(x, y)
clf.predict(x_test)