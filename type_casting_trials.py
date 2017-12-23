# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 20:00:19 2017

@author: arden
"""

import numpy as np
from sklearn import random_projection

#note that input is default cast to float64

#creating data list of 'float32' type to test sklearn transvormation on
rng = np.random.RandomState(0)
x = rng.rand(10, 2000)
x1 = np.array(x, dtype='float32') #bit transformation

transformer = random_projection.GaussianRandomProjection()
x_transform = transformer.fit_transform(x1)
print(x == x_transform)
