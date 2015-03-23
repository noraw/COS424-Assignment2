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
    array = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    array = np.reshape(array,(numofchromos,-1))
    return array



# argument parsing.
parser = argparse.ArgumentParser(description='Predict Genome.')
parser.add_argument("-M", "--MeanValue", action="store_true", help="run mean value prediction")
parser.add_argument("-R", "--LinearRegression", action="store_true", help="run linear regression")
parser.add_argument("-L", "--Lasso", action="store_true", help="run Lasso")
parser.add_argument("-G", "--Guassian", action="store_true", help="run Gaussian process")

# OTHER INPUT VARIABLES
outname = ""
inX0 = ""
inX1 = ""
inXU = ""
inX_test0 = ""
inX_test1 = ""
inX_testU = ""
inY = ""
inY_test = ""
numofchromos = 400

args = parser.parse_args()
print args;

X0 = read_dat_file(inX0, numofchromos) # matrix of training feature data (1s)
X1 = read_dat_file(inX1, numofchromos) # matrix of training feature data (1s)
XU = read_dat_file(inXU, numofchromos) # matrix of training feature data (1s)
X_test0 = read_dat_file(inX_test0, numofchromos) # matrix of testing feature data (0s)
X_test1 = read_dat_file(inX_test1, numofchromos) # matrix of testing feature data (0s)
X_testU = read_dat_file(inX_testU, numofchromos) # matrix of testing feature data (0s)
y = read_dat_file(inY, numofchromos) # vector of training values (1s)
y_test = read_dat_file(inY_test, numofchromos) # vector of testing ground truth values (0s)

print "X shape" + str(X.shape)
print "X test shape " + str(X_test.shape)
print "Y shape" + str(y.shape)
print "Y test shape " + str(y_test.shape)


# CLASSIFY!
if args.MeanValue:
    print "Mean Value"
    outname = "meanValue"

if args.LinearRegression:
    print "Linear Regression"
    outname = "linearRegression"
    clf = linear_model.LinearRegression()
    #LinearRegression(copy_X=True, fit_intercept=True, normalize=False)

if args.Lasso:
    print "Lasso"
    outname = "lasso"
    clf = linear_model.Lasso(alpha=.2)

if args.Gaussian:
    print "Gaussian"
    outname = "guassian"
    clf = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)

if args.MeanValue or args.LinearRegression or args.Lasso or args.Guassian:
    print "Nan = 0"
    feature_importance0 = predict(clf, X0, y, X_test0, y_test)

    print "Nan = 1"
    feature_importance1 = predict(clf, X1, y, X_test1, y_test)

    print "Nan = Uniform"
    feature_importanceU = predict(clf, XU, y, X_testU, y_test)

    # output feature importance for graphs
    outfile = file("feature_importance_" + outname + ".csv", "w")
    outfile.write('"Feature ID","0 Nans","1 Nans","Uniform Nans"\n');
    for i in range (len(feature_importance0)):
        outLine = str(i) + ","
        outLine += str(feature_importance0[i]) + ","
        outLine += str(feature_importance1[i]) + ","
        outLine += str(feature_importanceU[i])
        outLine += "\n"
        outfile.write(outLine)
    outfile.close();



def predict(clf, X, y, X_test, y_test):
    start = timeit.default_timer()
    clf.fit(X0, y)
    print "   predictions done.";
    predicted = clf.predict(X_test0)
    #probs_train = clf.predict_proba(X)    
    probs_test = clf.predict_proba(X_test0)
    print "   R2:      " + str(metrics.r2_score(y_test, predicted)) # classifier accuracy
    print "   RSME:    " + str(sqrt(metrics.mean_squared_error(y_test, predicted))) # classifier accuracy
    print(metrics.classification_report(y_test, predicted))
    stop = timeit.default_timer()
    print "   runtime: " + str(stop - start)
    return clf.coef_





