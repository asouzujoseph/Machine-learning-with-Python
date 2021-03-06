# -*- coding: utf-8 -*-
"""custom_Kfold-CV.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1B8yzBJQ2MRg2ZddK4WoudXB0e_5tIKVF
"""

import numpy as np
import pandas as pd
import random

## dataset
list1 = [10, 20, 30, 40, 50,60,70,80,90]  
vtr = np.array(list1)

def kfold (dataset, K):
  '''Split a dataset into k folds'''
  dataset_split = list()
  dataset_copy = list(dataset)
  fold_size = int(len(dataset) / K)
  for i in range(K):
	  fold = list()
	  while len(fold) < fold_size:
		  index = random.randrange(len(dataset_copy))
		  fold.append(dataset_copy.pop(index))
	  dataset_split.append(fold)
  return dataset_split

def CV(dataset,K):
  ''' function to pick one fold as validation set and use the rest of the k-folds as training set'''
  folds = list(range(0,K))
  CV_data = {}
  d = 0
  for i in folds:
    while d<len(folds):
      cc = folds.pop(d)
      validationSet = tuple(dataset[cc])
      trainingSet = dataset[folds[0]]+dataset[folds[1]]
      CV_data[validationSet] = trainingSet
      folds.insert(d,d)
      d += 1
  return CV_data

### example
grid_dataset  = kfold(vtr,3)
CV(grid_dataset,3)  ### produces a dictionary of the vaidation set as tuples and the training K-folds as a list.