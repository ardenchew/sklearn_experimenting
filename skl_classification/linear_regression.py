# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:34:31 2017

@author: arden
"""

import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


reg = linear_model.LinearRegression()
reg.fit([[0,0],[1,1],[2,2]],[0,1,2])
reg.coef_