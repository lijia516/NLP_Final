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

from sklearn.neighbors import KNeighborsClassifier
print "wsd class and function"
#WSD instance for a word associated with its sense, sentence and it position in a word
class myInstance:
    def __init__(self, sense, sentence, position):
        self.sense = sense
        self.sentence = sentence
        self.position = position

#split data into two parts
def split_senseval(file, cross_val):
            train_list = []
            label_list = []
            for i in range(cross_val):
                train_list.append(list())
                label_list.append(list())

            test = []
            total_counter = 0
            train_counter = 0
            for inst in nltk.corpus.senseval.instances(file):
                sentence = []
                offset = 0
                for i in range(len(inst.context)):
                    t = inst.context[i]
                    if type(t) == tuple:
                        sentence.append(t)
                    else:
                        if i < inst.position:
                            offset += 1
                senselabel = inst.senses[0]
                newInstance = myInstance(senselabel, sentence, inst.position - offset)
                total_counter += 1
                if (total_counter % 10):
                    index = train_counter % cross_val
                    train_list[index].append(newInstance)
                    label_list[index].append(senselabel)

                    train_counter += 1 # start from zero
                else: test.append(newInstance)
            print "total" + str(total_counter)
            return (train_list, label_list, test)

#feature extraction
def features_selection(instance, start, end):
    features = {} 
    word, pos = instance.sentence[instance.position]
    features[word] = True
    features[pos] = True
    for word, pos in instance.sentence:
        features[word] = True
    btm = instance.position - start if instance.position - start >= 0 else 0
    top = instance.position + end if instance.position + end < len(instance.sentence) else len(instance.sentence) - 1

    for i in range(btm,top):
        if instance.sentence[i] is not None:
            word, pos = instance.sentence[i]
            features[str(i - instance.position)] = pos
            features[str(i - instance.position)+"W"] = word
    return features

def create_cross_data(train_list, test, start, end):
    training_set = []
    testing_set = []
    cv_testing_set = []
    testing_set = [ (features_selection(i,start,end)) for i in test ]
    test_label = [ (i.sense) for i in test ]
    for j in range(len(train_list)):
        training_set.append([ (features_selection(i,start,end), i.sense) for i in train_list[j]])
        cv_testing_set.append([ (features_selection(i,start,end)) for i in train_list[j]])

    cv_training_set = []
    for j in range(len(training_set)):
        cv_training_set.append(list())
        for k in range(len(training_set)):
            if k == j:
                continue
            cv_training_set[j] += training_set[k]

    all_trn = []
    for k in range(len(training_set)):
        all_trn += training_set[k]

#    for j in range(len(cv_training_set)):
#        print len(cv_training_set[j])
    
    return (cv_training_set, cv_testing_set, all_trn, testing_set, test_label)
    
def __create_feature_set_from_cl(result_set_list):
    feature_set = []
    for i in range(len(result_set_list[0])):#case?
        feature_set.append(dict())
        for j in range(len(result_set_list)):#classifier?
            feature_set[i][j] = result_set_list[j][i]
    return feature_set

def __attach_label(feature_set, label):
    training_data = []
    for i in range(len(feature_set)):
        training_data.append((feature_set[i], label[i]))
    return training_data

def __seperate_cross_validation(feature_set_list, label_list):
    training_list = []
    for i in range(len(feature_set_list)):
        training_list.append(__attach_label(feature_set_list[i], label_list[i]))

    cv_trn = []
    for i in range(len(training_list)):
        cv_trn.append(list())
        for k in range(len(training_list)):
            if k == i:
                continue
            cv_trn[i] += training_list[k]

    all_trn = []
    for k in range(len(training_list)):
        all_trn += training_list[k]
    return (cv_trn, all_trn)

def create_meta_training_data(cl_result_list, label_list):
    feature_set_list = []
    for i in range(len(cl_result_list[0])):#how many cross_validation set
        tmp = []
        for k in range(len(cl_result_list)):#how many classifier
            tmp.append(cl_result_list[k][i])
        feature_set_list.append(__create_feature_set_from_cl(tmp))
    cv_trn, all_trn = __seperate_cross_validation(feature_set_list, label_list)
    return (cv_trn, feature_set_list, all_trn)

def merge_feature(feature_list):
    merge = []
    for i in range(len(feature_list[0])):
        merge.append(dict())
        for j in range(len(feature_list)):
            merge[i][j] = feature_list[j][i]
    return merge

def compute_accuracy(result, truth):
    counter = 0
    for i in range(len(result)):
        if result[i] == truth[i]:
            counter+=1
    return counter / float(len(result))

#cross = 4;
#print "data preprocessing"
#(train_list, label_list, test) = split_senseval('interest.pos',4)
#cv_trn, cv_t, all_trn, test_feature, test_label = create_cross_data(train_list, test)

#knn = SklearnClassifier(KNeighborsClassifier(3)).train(all_trn)
#print compute_accuracy(test_label, knn.classify_many(test_feature))
