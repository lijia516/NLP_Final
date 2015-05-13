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

#with open('serve.pickle') as f:
#with open('line.pickle') as f:
#with open('hard.pickle') as f:
#with open('interest.pickle') as f:
#    train_list, label_list, test = pickle.load(f)

#cv_trn, cv_t, all_trn, test_feature, test_label = create_cross_data(train_list, test)

#svm = SklearnClassifier(LinearSVC()).train(all_trn)
#svm_original = svm.classify_many(test_feature)
#print compute_accuracy(test_label, svm_original)
#dt = SklearnClassifier(tree.DecisionTreeClassifier()).train(all_trn)
#dt_original = dt.classify_many(test_feature)
#print compute_accuracy(test_label, dt_original)
#nb = nltk.NaiveBayesClassifier.train(all_trn)
#nb_original = nb.classify_many(test_feature)
#print compute_accuracy(test_label, nb_original)
#ent = nltk.classify.maxent.MaxentClassifier.train(all_trn, min_lldelta=0.05, max_iter=10)
#ent_original = ent.classify_many(test_feature)
#print compute_accuracy(test_label, ent_original)
#knn = SklearnClassifier(KNeighborsClassifier(3)).train(all_trn)
#knn_original = knn.classify_many(test_feature)
#print compute_accuracy(test_label, knn_original)
#
#with open('serve_original_acc.pickle', 'w') as f:
#with open('line_original_acc.pickle', 'w') as f:
#with open('hard_original_acc.pickle', 'w') as f:
#with open('interest_original_acc.pickle', 'w') as f:
#    pickle.dump([svm_original, dt_original, nb_original, ent_original, knn_original], f)
#
with open('serve_original_acc.pickle') as f:
    s1o, d1o, n1o, e1o, k1o = pickle.load(f)
with open('line_original_acc.pickle') as f:
    s2o, d2o, n2o, e2o, k2o = pickle.load(f)
with open('hard_original_acc.pickle') as f:
    s3o, d3o, n3o, e3o, k3o = pickle.load(f)
with open('interest_original_acc.pickle',) as f:
    s4o, d4o, n4o, e4o, k4o = pickle.load(f)

result_list = []
#result_list.append(s1o)
#result_list.append(d1o)
#result_list.append(n1o)
#result_list.append(e1o)
#result_list.append(k1o)
result_list.append(s1o + s2o + s3o + s4o)
result_list.append(d1o + d2o + d3o + d4o)
result_list.append(n1o + n2o + n3o + n4o)
result_list.append(e1o + e2o + e3o + e4o)
result_list.append(k1o + k2o + k3o + k4o)

label_all = []
with open('serve.pickle') as f:
    train_list, label_list, test = pickle.load(f)
cv_trn, cv_t, all_trn, test_feature, test_label = create_cross_data(train_list, test)
label_all += test_label
with open('line.pickle') as f:
    train_list, label_list, test = pickle.load(f)
cv_trn, cv_t, all_trn, test_feature, test_label = create_cross_data(train_list, test)
label_all += test_label
with open('hard.pickle') as f:
    train_list, label_list, test = pickle.load(f)
cv_trn, cv_t, all_trn, test_feature, test_label = create_cross_data(train_list, test)
label_all += test_label
with open('interest.pickle') as f:
    train_list, label_list, test = pickle.load(f)
cv_trn, cv_t, all_trn, test_feature, test_label = create_cross_data(train_list, test)
label_all += test_label

result_list.append(label_all)

for i in range(len(result_list)):
    for j in range(len(result_list)):
        print str(i)+"_"+str(j)
        print compute_accuracy(result_list[i],result_list[j])

