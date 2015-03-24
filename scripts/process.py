#!/usr/bin/python
import os
import numpy
import random

chr_no = 1
cutoff = 20

def process_nan(tokens, nan):
  data = []
  for token in tokens:
    if token == 'nan':
      if nan > 1:
        nan = random.random()
      x = float(nan)
    else:
      x = float(token)
    #print x
    data.append(x)
  #print data
  return data

def generate_dat(chr, cutoff, nan):
  train_filename = "/u/biancad/COS424/data/intersected_final_chr%d_cutoff_%d_train_revised.bed" %(chr_no, cutoff)
  truth_filename = "/u/biancad/COS424/data/intersected_final_chr%d_cutoff_%d_test.bed" %(chr_no, cutoff)
  #result_filename = "/u/biancad/COS424/data/intersected_final_chr%d_cutoff_%d_sample.bed" %(chr_no, cutoff)

  # parse the 32 sample values
  samples_train = []
  samples_test = []
  with open(train_filename) as train_file:
    lines = train_file.readlines()

    for line in lines:
      #print line
      tokens = line.split()
      #print tokens, len(tokens)
      samples = process_nan(tokens[4:37],nan)
      #print samples, len(samples)
      if tokens[37] == '1':
        samples_train.append(samples)
      else:
        samples_test.append(samples)

    print len(samples_train),"trains", len(samples_test), "tests"

    matrix_train  = numpy.zeros(shape=(len(samples_train),32), dtype=numpy.uint8)
    for i in range(len(samples_train)):
      for j in range(32):
        matrix_train[i,j] = samples_train[i][j]

    matrix_test = numpy.zeros(shape=(len(samples_test),32), dtype=numpy.uint8)
    for i in range(len(samples_test)):
      for j in range(32):
        matrix_test[i,j] = samples_test[i][j]
    
    if nan > 1:
      matrix_train.tofile("../data/XU_chr%d_cutoff_%d.dat" %(chr_no,cutoff))
      matrix_test.tofile("../data/X_testU_chr%d_cutoff_%d.dat" %(chr_no,cutoff))
    else:
      matrix_train.tofile("../data/X%d_chr%d_cutoff_%d.dat" %(nan,chr_no,cutoff))
      matrix_test.tofile("../data/X_test%d_chr%d_cutoff_%d.dat" %(nan,chr_no,cutoff))

  # parse the ground truth
  truth_train = []
  truth_test = []
  with open(truth_filename) as truth_file:
    lines = truth_file.readlines()

    for line in lines:
      #print line
      tokens = line.split()
      #print tokens, len(tokens)
      truth = process_nan(tokens[4:5],nan)
      #print truth, len(truth)
      if tokens[5] == '1':
        truth_train.append(truth)
      else:
        truth_test.append(truth)

    print len(truth_train),"trains", len(truth_test), "tests"
        
    vector_train  = numpy.zeros(shape=(len(truth_train),1), dtype=numpy.uint8)
    for i in range(len(truth_train)):
      for j in range(1):
        vector_train[i,j] = truth_train[i][j]

    vector_test = numpy.zeros(shape=(len(truth_test),1), dtype=numpy.uint8)
    for i in range(len(truth_test)):
      for j in range(1):
        vector_test[i,j] = truth_test[i][j]
        
    if nan > 1:
      vector_train.tofile("../data/yU_chr%d_cutoff_%d.dat" %(chr_no,cutoff))
      vector_test.tofile("../data/y_testU_chr%d_cutoff_%d.dat" %(chr_no,cutoff))
    else:
      vector_train.tofile("../data/y%d_chr%d_cutoff_%d.dat" %(nan,chr_no,cutoff))
      vector_test.tofile("../data/y_test%d_chr%d_cutoff_%d.dat" %(nan,chr_no,cutoff))


generate_dat(chr_no, cutoff, 0)
generate_dat(chr_no, cutoff, 1)
generate_dat(chr_no, cutoff, 2) # this is a hack for random

