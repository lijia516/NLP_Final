#!/usr/bin/env python
import nltk
from nltk.classify.scikitlearn import SklearnClassifier

#SVM
from sklearn.svm import LinearSVC

#DT
from sklearn import tree

#NN ?
from sklearn.neighbors import KDTree
import numpy as np

#stocastic gradient descent
from sklearn.linear_model import SGDClassifier

#save variable
import pickle
print "Data Preprocessing"

(train_list_s, label_list_s, test_s) = split_senseval('serve.pos',4)
(train_list_l, label_list_l, test_l) = split_senseval('line.pos',4)
(train_list_h, label_list_h, test_h) = split_senseval('hard.pos',4)
(train_list_i, label_list_i, test_i) = split_senseval('interest.pos',4)

with open('serve_r.pickle', 'w') as f:
    pickle.dump([train_list_s, label_list_s, test_s], f)

with open('line_r.pickle', 'w') as f:
    pickle.dump([train_list_l, label_list_l, test_l], f)

with open('hard_r.pickle', 'w') as f:
    pickle.dump([train_list_h, label_list_h, test_h], f)

with open('interest_r.pickle', 'w') as f:
    pickle.dump([train_list_i, label_list_i, test_i], f)
