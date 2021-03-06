# ********************************************
# Author: Nora Coler
# Date: Mar 24, 2015
#
#
# ********************************************
import numpy as np
import argparse
import os
from sklearn import metrics
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn.pipeline import Pipeline
from scipy import sparse
import timeit
from math import sqrt

# read a dat file back into python. 
def read_dat_file(myfile, numofchromos):
    array = np.fromfile(myfile, dtype=np.float64, count=-1, sep="")
    array = np.reshape(array,(numofchromos,-1))
    return array



def predict(clf, X, y, X_test, y_test):
    start = timeit.default_timer()
    clf.fit(X, y)
    print "   predictions done.";
    predicted = clf.predict(X_test)
    #probs_train = clf.predict_proba(X)    
    #probs_test = clf.predict_proba(X_test0)
    print "   R2:      " + str(metrics.r2_score(y_test, predicted)) # classifier accuracy
    print "   RSME:    " + str(sqrt(metrics.mean_squared_error(y_test, predicted))) # classifier accuracy
    #print(metrics.classification_report(y_test, predicted))
    stop = timeit.default_timer()
    print "   runtime: " + str(stop - start)

    if args.LinearRegression:
        return clf.coef_[0]
    if args.Lasso:
        return clf.coef_
    if args.RANSAC:
        return clf.estimator_.coef_[0]





# argument parsing.
parser = argparse.ArgumentParser(description='Predict Genome.')
parser.add_argument("-R", "--LinearRegression", action="store_true", help="run linear regression")
parser.add_argument("-L", "--Lasso", action="store_true", help="run Lasso")
parser.add_argument("-S", "--RANSAC", action="store_true", help="run RANSAC process")

# OTHER INPUT VARIABLES
outname = "" # assigned later
inX0 = "../data/X0_chr1_cutoff_20_len_7523.dat"
inXA = "../data/XA_chr1_cutoff_20_len_7523.dat"
inXR = "../data/XR_chr1_cutoff_20_len_7523.dat"
inX_test0 = "../data/X_test0_chr1_cutoff_20_len_368411.dat"
inX_testA = "../data/X_testA_chr1_cutoff_20_len_368411.dat"
inX_testR = "../data/X_testR_chr1_cutoff_20_len_368411.dat"
inY = "../data/y_chr1_cutoff_20_len_7523.dat"
inY_test = "../data/y_test_chr1_cutoff_20_len_368411.dat"
numofchromos = 7523
numofchromosTest = 368411

args = parser.parse_args()
print args;

X0 = read_dat_file(inX0, numofchromos) # matrix of training feature data (1s)
XA = read_dat_file(inXA, numofchromos) # matrix of training feature data (1s)
XR = read_dat_file(inXR, numofchromos) # matrix of training feature data (1s)
X_test0 = read_dat_file(inX_test0, numofchromosTest) # matrix of testing feature data (0s)
X_testA = read_dat_file(inX_testA, numofchromosTest) # matrix of testing feature data (0s)
X_testR = read_dat_file(inX_testR, numofchromosTest) # matrix of testing feature data (0s)
y = read_dat_file(inY, numofchromos) # vector of training values (1s)
y_test = read_dat_file(inY_test, numofchromosTest) # vector of testing ground truth values (0s)

print "X shape" + str(X0.shape)
print "X test shape " + str(X_test0.shape)
print "Y shape" + str(y.shape)
print "Y test shape " + str(y_test.shape)


# CLASSIFY!
if args.LinearRegression:
    print "Linear Regression"
    outname = "linearRegression"
    clf = linear_model.LinearRegression()
    #LinearRegression(copy_X=True, fit_intercept=True, normalize=False)

if args.Lasso:
    alphaIn=5.0
    print "Lasso: " + str(alphaIn)
    outname = "lasso"+str(alphaIn)
    clf = linear_model.Lasso(alpha=alphaIn)

if args.RANSAC:
    print "RANSAC"
    outname = "ransac"
    clf = linear_model.RANSACRegressor(linear_model.LinearRegression())


if args.LinearRegression or args.Lasso or args.RANSAC:
    print "Nan = 0"
    feature_importance0 = predict(clf, X0, y, X_test0, y_test)

    print "Nan = Average"
    feature_importanceA = predict(clf, XA, y, X_testA, y_test)

    print "Nan = Random"
    feature_importanceR = predict(clf, XR, y, X_testR, y_test)

    # output feature importance for graphs
    outfile = file("feature_importance_" + outname + ".csv", "w")
    outfile.write('"Feature ID","0 Nans","Average Nans","Random Nans"\n');
    print len(feature_importance0)
    for i in range (len(feature_importance0)):
        outLine = str(i) + ","
        outLine += str(feature_importance0[i]) + ","
        outLine += str(feature_importanceA[i]) + ","
        outLine += str(feature_importanceR[i])
        outLine += "\n"
        outfile.write(outLine)
    outfile.close();





