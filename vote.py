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

#

#load data
with open('interest.pickle') as f:
#with open('hard.pickle') as f:
#with open('line.pickle') as f:
#with open('serve.pickle') as f:
    train_list, label_list, test = pickle.load(f)

cv_trn, cv_t, all_trn, test_feature, test_label = create_cross_data(train_list, test)

svm_base = []
dt_base = []
nb_base = []
ent_base = []
result = []
svm_result = []
nb_result = []
dt_result = []
ent_result = []

svm_base = SklearnClassifier(LinearSVC()).train(all_trn)
dt_base = SklearnClassifier(tree.DecisionTreeClassifier()).train(all_trn)
nb_base = nltk.NaiveBayesClassifier.train(all_trn)
ent_base = nltk.classify.maxent.MaxentClassifier.train(all_trn, trace=1, max_iter=4)
knn_base = SklearnClassifier(KNeighborsClassifier(5)).train(all_trn)

#test
intermediate = []
intermediate.append(svm_base.classify_many(test_feature))
intermediate.append(dt_base.classify_many(test_feature))
intermediate.append(nb_base.classify_many(test_feature))
intermediate.append(ent_base.classify_many(test_feature))
intermediate.append(knn_base.classify_many(test_feature))
#final_feature = merge_feature(intermediate)

#print compute_accuracy(test_label, svm_meta.classify_many(final_feature))
#print compute_accuracy(test_label, max_ent_meta.classify_many(final_feature))

#Weight:
weight = []
weight.append(0.9)
weight.append(0.8)
weight.append(0.8)
weight.append(0.8)
weight.append(0.6)

voteResult = []
voteResultNW = []
#weighted voting
for i in range(len(intermediate[0])):#for each case
    thisResult = {}
    thisResultNW = {}
    for j in range(len(intermediate)):
        vote = intermediate[j][i]
        if vote in thisResult:
            thisResult[vote] += weight[j]
            thisResultNW[vote] += 1;
        else:
            thisResult[vote] = weight[j]
            thisResultNW[vote] = 1;
    print thisResult
    print thisResultNW
    count = 0
    for key, value in thisResult.iteritems():
        if value > count:
            vote = key
            count = value
    voteResult.append(vote)
    count = 0
    for key, value in thisResultNW.iteritems():
        if value >= count:
            vote = key
            count = value
    voteResultNW.append(vote)
print len(voteResultNW)
print compute_accuracy(test_label, voteResult)
print compute_accuracy(test_label, voteResultNW)
