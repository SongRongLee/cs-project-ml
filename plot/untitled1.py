# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 20:08:58 2018

@author: Administrator
"""

import plotly.offline as py
import plotly.graph_objs as go
import os
import pandas as pd
text_file = pd.read_csv(os.path.dirname(__file__)+"/matrix/matrix0.txt", delimiter=" ", header=None).dropna(axis=1)
z=[[1, 20, 30],
                      [20, 1, 60],
                      [30, 60, 1]]
qq=text_file.values.tolist()