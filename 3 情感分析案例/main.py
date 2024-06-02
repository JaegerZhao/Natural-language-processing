#!/usr/bin/python

import random
import collections
import math
import sys
from util import *
from model import *

def TestModel(numIters, eta):
    trainExamples = readExamples('data/data_rt.train')
    testExamples = readExamples('data/data_rt.test')
    featureExtractor = extractFeatures
    weights = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=numIters, eta=eta)
    trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    testError = evaluatePredictor(testExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print ("train error = %s, test error = %s" % (trainError, testError))

if __name__ == "__main__":
    TestModel(20, 0.01)