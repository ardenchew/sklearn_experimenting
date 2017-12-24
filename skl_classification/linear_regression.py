# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:34:31 2017

@author: arden
"""

import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


diabetes = datasets.load_diabetes()

