#!/usr/bin/env python
import nltk
from nltk.classify.scikitlearn import SklearnClassifier

#SVM
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

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
    train_list, label_list, test = pickle.load(f)

cv_trn, cv_t, all_trn, test_feature, test_label = create_cross_data(train_list, test)

svm_base = []
dt_base = []
nb_base = []
ent_base = []
knn_base = []
result = []
svm_result = []
nb_result = []
dt_result = []
ent_result = []
knn_result = []
for i in range(len(cv_trn)):
    print "layer_1_cv:"+str(i)
    svm_base.append(SklearnClassifier(LinearSVC()).train(cv_trn[i]))
    dt_base.append(SklearnClassifier(tree.DecisionTreeClassifier()).train(cv_trn[i]))
    ent_base.append(nltk.classify.maxent.MaxentClassifier.train(cv_trn[i], trace=1, max_iter=4))
    nb_base.append(nltk.NaiveBayesClassifier.train(cv_trn[i]))
    knn_base.append(SklearnClassifier(KNeighborsClassifier(5)).train(cv_trn[i]))

    svm_result.append(svm_base[i].classify_many(cv_t[i]))
    dt_result.append(dt_base[i].classify_many(cv_t[i]))
    ent_result.append(ent_base[i].classify_many(cv_t[i]))
    nb_result.append(nb_base[i].classify_many(cv_t[i]))
    knn_result.append(knn_base[i].classify_many(cv_t[i]))

#base classifiers
svm_base = SklearnClassifier(LinearSVC()).train(all_trn)
dt_base = SklearnClassifier(tree.DecisionTreeClassifier()).train(all_trn)
ent_base = nltk.classify.maxent.MaxentClassifier.train(all_trn, trace=1, max_iter=4)
nb_base = nltk.NaiveBayesClassifier.train(all_trn)
knn_base = SklearnClassifier(KNeighborsClassifier(5)).train(all_trn)

result.append(svm_result)
result.append(dt_result)
result.append(ent_result)
result.append(nb_result)
result.append(knn_result)

#meta cl
cv_trn2, cv_t2, all_trn2 = create_meta_training_data(result, label_list)
svm_meta = SklearnClassifier(LinearSVC()).train(all_trn2)
dt_meta = SklearnClassifier(tree.DecisionTreeClassifier()).train(all_trn2)
ent_meta = nltk.classify.maxent.MaxentClassifier.train(all_trn2, trace=1, max_iter=4)
nb_meta = nltk.NaiveBayesClassifier.train(all_trn2)
knn_meta = SklearnClassifier(KNeighborsClassifier(5)).train(all_trn2)

svm_l2 = []
dt_l2 = []
nb_l2 = []
ent_l2 = []
knn_l2 = []
result = []
svm_result = []
nb_result = []
dt_result = []
ent_result = []
knn_result = []

for i in range(len(cv_trn)):
    print "layer_2_cv:"+str(i)
    svm_l2.append(SklearnClassifier(LinearSVC()).train(cv_trn2[i]))
    dt_l2.append(SklearnClassifier(tree.DecisionTreeClassifier()).train(cv_trn2[i]))
    ent_l2.append(nltk.classify.maxent.MaxentClassifier.train(cv_trn2[i], trace=1, max_iter=4))
    nb_l2.append(nltk.NaiveBayesClassifier.train(cv_trn2[i]))
    knn_l2.append(SklearnClassifier(KNeighborsClassifier(5)).train(cv_trn2[i]))

    svm_result.append(svm_l2[i].classify_many(cv_t2[i]))
    dt_result.append(dt_l2[i].classify_many(cv_t2[i]))
    ent_result.append(ent_l2[i].classify_many(cv_t2[i]))
    nb_result.append(nb_l2[i].classify_many(cv_t2[i]))
    knn_result.append(knn_l2[i].classify_many(cv_t2[i]))

result.append(svm_result)
result.append(dt_result)
result.append(ent_result)
result.append(nb_result)
result.append(knn_result)

cv_trn3, cv_t3, all_trn3 = create_meta_training_data(result, label_list)
svm_meta_meta = SklearnClassifier(LinearSVC()).train(all_trn3)

#test 1-layer
intermediate = []
intermediate.append(svm_base.classify_many(test_feature))
intermediate.append(dt_base.classify_many(test_feature))
intermediate.append(nb_base.classify_many(test_feature))
intermediate.append(ent_base.classify_many(test_feature))
intermediate.append(knn_base.classify_many(test_feature))
test_feature = merge_feature(intermediate)

#test 1-layer
intermediate = []
intermediate.append(svm_meta.classify_many(test_feature))
intermediate.append(dt_meta.classify_many(test_feature))
intermediate.append(nb_meta.classify_many(test_feature))
intermediate.append(ent_meta.classify_many(test_feature))
intermediate.append(knn_meta.classify_many(test_feature))

final_feature = merge_feature(intermediate)

print compute_accuracy(test_label, svm_meta_meta.classify_many(final_feature))
#print compute_accuracy(test_label, max_ent_meta.classify_many(final_feature))
