# -*- coding: utf-8 -*-
"""
Created on Thu Oct 04 13:59:00 2018

@author: Benli
"""

import os
import sys
import pandas as pd
from sklearn.cross_validation import train_test_split

gpcr_name = sys.argv[1]
gpcr_length = sys.argv[2]
gpcr_radius = sys.argv[3]

gpcr_diameter = int(gpcr_radius) * 2
file_name = gpcr_name + '_ECFP' + str(gpcr_diameter) + '_' + gpcr_length + '_Top300.csv'
local_path = os.path.dirname(os.getcwd())
gpcr_path = local_path +'\\data\\'+ gpcr_name
DNN_path = local_path + '\\DeepNeuralNet-QSAR\\'
DNN_data_path = DNN_path + '\\' + gpcr_name

if not os.path.exists(DNN_data_path):
    os.makedirs(DNN_data_path)

feature = pd.read_csv(file_name,header=None)
response = pd.read_csv(gpcr_path + '\\Response.csv',header=None)

X_train, X_test, y_train, y_test = train_test_split(feature, response, random_state=1)

col = []
for i in X_train.columns:
    col.append('D_' + str(i))

ind = []
for i in X_train.index:
    ind.append('M_' + str(i))

X_train.columns = col
X_train.index = ind

y_train.columns = ['Act']
y_train.index = ind
d_train = pd.concat([y_train, X_train], axis=1)
d_train.index.name = 'MOLECULE'
d_train.to_csv(DNN_data_path + '\\' + 'ECFP'+ str(gpcr_diameter) + '_' + gpcr_length + '_training.csv',sep=',')

col = []
for i in X_test.columns:
    col.append('D_' + str(i))

ind = []
for i in X_test.index:
    ind.append('M_' + str(i))

X_test.columns = col
X_test.index = ind

y_test.columns = ['Act']
y_test.index = ind
d_test = pd.concat([y_test, X_test], axis=1)
d_test.index.name = 'MOLECULE'
d_test.to_csv(DNN_data_path + '\\' + 'ECFP'+ str(gpcr_diameter) + '_' + gpcr_length + '_test.csv',sep=',')