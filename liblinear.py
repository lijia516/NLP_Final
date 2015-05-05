
import nltk

import tempfile
import shutil
import sys
import os
import math
import array

import platform

if platform.system() == "Linux":
    llname = "./ll-train_linux"
elif platform.system() == "Windows":
    llname = "ll-train_win"
else:
    # Always "Darwin" for a Mac?
    llname = "./ll-train_mac"

class Constant(nltk.classify.api.ClassifierI):    
    def __init__(self, c):
        self.c = c
    def labels(self):
        return [self.c]
    def classify(self, featureset):
        return self.c

    @staticmethod
    def train(training_set):
        return Constant('l')

class FeatureDictionary:
    def __init__(self, normalize=True, presence=True):
        self.dict = {}
        self.frozen = False
        self.normalize = normalize
        self.presence = presence
    def freeze(self):
        self.frozen = True
    def size(self):
        return len(self.dict)
    def encode(self, symbol):
        enc = self.dict.get(symbol)
        if enc:
            return enc
        if self.frozen:
            return None
        val = len(self.dict) + 1
        self.dict[symbol] = val
        return val
    def encode_bag_of_words(self, words):
        h = {}
        for w in words:
            h[w] = h.get(w, 0) + 1
        arr = []
        sqsum = 0.0
        for p in h.iteritems():
            k = self.encode(p[0])
            if k:
                if self.presence:
                    arr.append((k, 1.0))
                    sqsum = sqsum + 1.0
                else:
                    arr.append((k, 1.0 * p[1]))
                    sqsum = sqsum + p[1]*p[1]
        out = sorted(arr, key=lambda p: p[0])
        if self.normalize and sqsum > 0.0:
            factor = 1.0 / math.sqrt(sqsum)
            return [(k, v*factor) for (k, v) in out]
        else:
            return out
    def encode_attrval(self, attr, val):
        return self.encode(repr(attr) + " = " + repr(val))

    def encode_feats(self, fd):
        arr = []
        for p in fd.iteritems():
            k = self.encode_attrval(p[0], p[1])
            if k:
                arr.append((k, 1.0))
        out = sorted(arr, key=lambda p: p[0])
        if self.normalize and len(out) > 0:
            factor = 1.0 / math.sqrt(len(out))
            return [(k, v*factor) for (k, v) in out]
        else:
            return out

    def encode_general(self, obj):
        if type(obj) == dict:
            return self.encode_feats(obj)
        if type(obj) == list:
            f0 = obj[0]
            if type(f0) == str or type(f0) == unicode:
                return self.encode_bag_of_words(obj)
            if type(f0) == tuple:
                if type(f0[0]) == int:
                    return obj
        raise ValueError("unknown feature type")

    def inverse(self):
        out = [None] * (self.size() + 1)
        for p in self.dict.iteritems():
            out[p[1]] = p[0]
        return out

    def decode(self, e):
        for (k, v) in self.dict.iteritems():
            if v == e:
                return k

    @staticmethod
    def sv_to_str(sv):
        out = ""
        for p in sv:
            if out != "":
                out = out + " "
            out = out + str(p[0]) + ":" + str(p[1])
        return out

class BinaryLinearClassifier(nltk.classify.api.ClassifierI):    
    def __init__(self, fenc, cldec, w, bias, labels_enc):
        self.fenc = fenc
        self.w = w
        self.bias = bias
        self.lpos = cldec[labels_enc[0]]
        self.lneg = cldec[labels_enc[1]]
    def labels(self):
        return [self.lpos, self.lneg]
    def score(self, featureset):
        score = self.bias
        fv = self.fenc.encode_general(featureset)
        for (i, v) in fv:
            score = score + v*self.w[i]
        return score
    def classify(self, featureset):
        score = self.score(featureset)
        if(score > 0):
            return self.lpos
        else:
            return self.lneg        
    def prob_classify(self, featureset):
        score = self.score(featureset)
        pr = 1.0/(1.0 + math.exp(-score))
        return nltk.probability.DictionaryProbDist({self.lpos:pr, 
                                                    self.lneg:(1.0-pr)})
    def rank_features(self):
        lp = []
        ln = []
        for fid in xrange(1, self.fenc.size()+1):
            score = self.score([(fid, 1.0)])
            pr = 1.0/(1.0 + math.exp(-score))
            if pr > 0.5:
                ratio = pr / (1.0 - pr)
                lp.append( (fid, self.lpos, self.lneg, ratio) )
            else:
                ratio = (1.0 - pr) / pr
                ln.append( (fid, self.lneg, self.lpos, ratio) )
        return (sorted(lp, key=lambda t: -t[3]), sorted(ln, key=lambda t: -t[3]))
    def show_most_informative_features(self, n=10):
        (lp, ln) = self.rank_features()        
        print "* Strongest features for " + str(self.lpos) + ":"
        for (fi, l1, l2, r) in lp[:n]:
            fl = self.fenc.decode(fi)
            print '{0:15} {1} : {2}     {3:.3f} : 1.0'.format(fl, l1, l2, r)
        print "* Strongest features for " + str(self.lneg) + ":"
        for (fi, l1, l2, r) in ln[:n]:
            fl = self.fenc.decode(fi)
            print '{0:15} {1} : {2}     {3:.3f} : 1.0'.format(fl, l1, l2, r)

class MulticlassLinearClassifier(nltk.classify.api.ClassifierI):    
    def __init__(self, fenc, cldec, ws, biases, labels_enc):
        self.fenc = fenc
        self.cldec = cldec
        self.ws = ws
        self.biases = biases
        self.labels_enc = labels_enc
        self.nclasses = len(labels_enc)
    def labels(self):
        return cldec[1:]
    def classify(self, featureset):
        fv = self.fenc.encode_general(featureset)
        maxscore = -1e100 # inf in python?
        maxindex = -1
        for j in xrange(0, self.nclasses):
            score = self.biases[j]            
            for (i, v) in fv:
                score = score + v*self.ws[j][i]
            if score > maxscore:
                maxscore = score
                maxindex = j
        return self.cldec[self.labels_enc[maxindex]]
    def prob_classify(self, featureset):
        fv = self.fenc.encode_general(featureset)
        pd = {}
        for j in xrange(0, self.nclasses):
            score = self.biases[j]
            for (i, v) in fv:
                score = score + v*self.ws[j][i]
            l = self.cldec[self.labels_enc[j]]
            score = math.exp(score)
            pd[l] = score
        return nltk.probability.DictionaryProbDist(pd, normalize=True)
    def rank_features(self):
        flist = []
        for fid in xrange(1, self.fenc.size()+1):
            fv = [(fid, 1.0)]
            maxscore = -1e100 # inf in python?
            maxl = -1
            maxscore2 = -1e100
            maxl2 = -1
            pd = self.prob_classify(fv)            
            for j in xrange(0, self.nclasses):
                l = self.cldec[self.labels_enc[j]]
                score = pd.prob(l)
                if score > maxscore:
                    maxscore2 = maxscore
                    maxl2 = maxl
                    maxscore = score
                    maxl = l
                elif score > maxscore2:
                    maxscore2 = score
                    maxl2 = l
                    
            ratio = maxscore / maxscore2
            flist.append( (fid, maxl, maxl2, ratio) )
        return sorted(flist, key=lambda t: -t[3])
    def show_most_informative_features(self, n=10):
        l = self.rank_features()[:n]
        for (fi, l1, l2, r) in l:
            fl = self.fenc.decode(fi)
            print '{0:15} {1} : {2}     {3:.3f} : 1.0'.format(fl, l1, l2, r)

class LibLinear(nltk.classify.api.ClassifierI):
    def __init__(self):
        raise Exception("cannot be initialized")

    @staticmethod
    def read_binary_classifier(filename):
        bias = None
        nr_feature = None
        nr_class = None
        labels_enc = None
        
        f = open(filename)
        
        for line in f:
            line = line.strip()
            #print "|" + line + "|"
            if line == "w":
                break
            elif line.startswith("bias"):
                bias = float(line[4:])
            elif line.startswith("nr_feature"):
                nr_feature = int(line[10:])
            elif line.startswith("nr_class"):
                nr_class = int(line[8:])
            elif line.startswith("label"):
                labels_enc = map(int, line[6:].split())

        index = 0
        w = array.array('d', [0.0] * (nr_feature + 1))
        bias_weight = 0.0
        for line in f:
            line = line.strip()
            #print "|" + line + "|"
            d = float(line)
            #print "|" + str(d) + "|"
            if index == nr_feature and bias >= 0.0:
                bias_weight = d            
            elif index >= nr_feature:
                raise Exception("illegal format of classifier")
            else:
                w[index + 1] = d
            index = index + 1
        if (bias < 0 and index < nr_feature) or (bias >= 0 and index <= nr_feature):
            raise Exception("illegal format of classifier")
        f.close()
        return (w, bias*bias_weight, labels_enc)

    @staticmethod
    def read_multiclass_classifier(filename):
        bias = None
        nr_feature = None
        nr_class = None
        labels_enc = None
        
        f = open(filename)
        
        for line in f:
            line = line.strip()
            #print "|" + line + "|"
            if line == "w":
                break
            elif line.startswith("bias"):
                bias = float(line[4:])
            elif line.startswith("nr_feature"):
                nr_feature = int(line[10:])
            elif line.startswith("nr_class"):
                nr_class = int(line[8:])
            elif line.startswith("label"):
                labels_enc = map(int, line[6:].split())

        index = 0
        w = []
        for i in xrange(0, nr_class):
            w.append(array.array('d', [0.0] * (nr_feature + 1)))
        bias_weights = [0.0] * nr_class
        for line in f:
            line = line.strip()
            #print line.split(" ")
            ds = map(float, line.split(" "))
            if index == nr_feature and bias >= 0.0:
                bias_weights = ds
            elif index >= nr_feature:
                raise Exception("illegal format of classifier")
            else:
                for i in xrange(0, nr_class):
                    w[i][index + 1] = ds[i]
            index = index + 1
        if (bias < 0 and index < nr_feature) or (bias >= 0 and index <= nr_feature):
            raise Exception("illegal format of classifier")
        f.close()
        for i in xrange(0, nr_class):
            bias_weights[i] = bias_weights[i] * bias
        return (w, bias_weights, labels_enc)

    @staticmethod
    def get_feature_type(training_set):
        # 1: attr-val dictionary
        # 2: bag of words
        # 3: sparse vector
        (feats, val) = training_set[0]
        if type(feats) == dict:
            return 1 
        if type(feats) == list:
            f0 = feats[0]
            if type(f0) == str or type(f0) == unicode:
                return 2
            if type(f0) == tuple:
                if type(f0[0]) == int:
                    return 3
        return 0

    @staticmethod
    def train(training_set, fd=None, cld=None, s=-1, c=1, B=-1, quiet=True,
              param="",
              normalize=True, presence=True):
        ft = LibLinear.get_feature_type(training_set)
        if ft == 0:
            raise ValueError("illegal feature type in training set")
            
        dirname = tempfile.mkdtemp(prefix="ll_workdir")
        inputfile = os.path.join(dirname, "ll_input")
        outputfile = os.path.join(dirname, "ll_output")

        if not fd:
            fd = FeatureDictionary(normalize=normalize, presence=presence)
        if not cld:
            cld = FeatureDictionary()

        to_input = open(inputfile, "w")
        
        for (feats, val) in training_set:
            cl = cld.encode(val)
            if ft == 1:
                fs = fd.encode_feats(feats)
            elif ft == 2:
                fs = fd.encode_bag_of_words(feats)
            elif ft == 3:
                fs = feats
            else:
                raise ValueError("fs?")
            to_input.write(str(cl));
            to_input.write(" ");
            to_input.write(FeatureDictionary.sv_to_str(fs))
            to_input.write("\n")
        to_input.close()

        cld.freeze()
        fd.freeze()

        if not quiet:
            print "Feature dictionary size:", fd.size()

        labels = cld.inverse()
        
        if cld.size() == 0:
            return Constant(None)
        elif cld.size() == 1:
            return Constant(labels[1])

        if s < 0:
            if cld.size() == 2:
                s = 1
            elif cld.size() > 2:
                s = 4

        if cld.size() == 3 and s != 4:
            raise ValueError("s must be 4 for multiclass classifiers")
        #if cld.size() < 3 and s == 4:
        #    raise ValueError("s can be 4 for multiclass classifiers only")

        if quiet:
            qstr = " -q"
        else:
            qstr = ""
        command = llname + " -s " + str(s) + " -c " + str(c) + " -B " \
            + str(B) + qstr + " " + param + " " + inputfile + " " + outputfile
        if not quiet:
            print "Running |" + command + "|"

        retval = os.system(command)
        #if retval != 0:
        #    if not quiet:
        #        print "Running in local directory: |./" + command + "|"
        #    retval = os.system("./" + command)

        if not quiet:
            print "retval =", retval
        if retval != 0:
            print "Training failed!"
            shutil.rmtree(dirname, ignore_errors=True)
            return None

        if cld.size() == 2 and s != 4:
            t = LibLinear.read_binary_classifier(outputfile)
            shutil.rmtree(dirname, ignore_errors=True)
            return BinaryLinearClassifier(fd, labels, t[0], t[1], t[2])
        elif cld.size() > 2 or s == 4:
            t = LibLinear.read_multiclass_classifier(outputfile)
            shutil.rmtree(dirname, ignore_errors=True)
            return MulticlassLinearClassifier(fd, labels, t[0], t[1], t[2])

