import lstm as lstm
import Randomforest as rdf
import LightGBM as lgbm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('./train/data.csv')
with open('./train/tuple_process.pkl', 'rb') as f:
    tuple_process = pickle.load(f)

for t in tuple_process[1:]:
    country = t[0]
    feature = t[1]
    try:
        #lstm.predict(data,country,feature,1)
        #rdf.randomforest_func(data,country,feature,1)
        lgbm.lightgbm_func(data,country,feature,1)
    except:
        continue