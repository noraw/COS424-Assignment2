#!/usr/bin/python
import os
import numpy
import random

chr_no = 1
cutoff = 20
sample_size = 33

def process_nan(tokens, code):
  # array of floats with the nan samples resolved
  data = []
  #print tokens

  if code == '0':
    for token in tokens:
      if token == 'nan':
        x = float(0)
      else:
        x = float(token)
      #print x
      data.append(x)

  elif code == 'A':
    for i in range(len(tokens)):
      if tokens[i] == 'nan':
        # when the nan is the first sample 
        if i == 0:
          tokens[i] = float(0)
        # when the nan is the last sample 
        elif i == len(tokens)-1:
          tokens[i] = float(tokens[i-1])/2
        # when the nan is not the first sample or the last one, average with neighbour
        else:
          if tokens[i+1] == 'nan':
            tokens[i] = float(tokens[i-1])/2
          else:
            tokens[i] = (float(tokens[i-1]) + float(tokens[i+1]))/2
        x = tokens[i]
      else:
        x = float(tokens[i])
      data.append(x)

  elif code == 'R':
    max = float(-1)
    min = float(2)
    # figure out the min and max of the non-nan values
    for token in tokens:
      if not token == 'nan':
        x = float(token)
        if x > max:
          max = x
        if x < min:
          min = x
    #print min, max
    # assign all nan to uniform random number between min and max
    for token in tokens:
      if token == 'nan':
        x = random.uniform(min,max)
      else:
        x = float(token)
      data.append(x)

  else:
    for token in tokens:
      if token == 'nan':
        x = float(1)
      else:
        x = float(token)
      data.append(x)

  #print data
  return data


def generate_dat(chr, cutoff, nan_code):
  samples_filename = "/u/biancad/COS424/data/intersected_final_chr%d_cutoff_%d_train_revised.bed" %(chr_no, cutoff)
  truth_filename = "/u/biancad/COS424/data/intersected_final_chr%d_cutoff_%d_test.bed" %(chr_no, cutoff)
  #result_filename = "/u/biancad/COS424/data/intersected_final_chr%d_cutoff_%d_sample.bed" %(chr_no, cutoff)
  samples_file = open(samples_filename)
  truth_file = open(truth_filename)

  # parse the 33 sample values
  samples_train = []
  samples_test = []
  samples_lines = samples_file.readlines()
  # parse the ground truth
  truth_train = []
  truth_test = []
  truth_lines = truth_file.readlines()
  
  samples_distances_lines = []

  for i in range(len(samples_lines)):
    tokens = samples_lines[i].split()
    samples = process_nan(tokens[4:37],nan_code)
    if i == 0:
      prev_distance = float(0)
      next_distance = float(samples_lines[i+1].split()[1]) - float(tokens[1]) 
    elif i == len(samples_lines)-1:
      prev_distance = float(tokens[1]) - float(samples_lines[i-1].split()[1]) 
      next_distance = float(0)
    else:
      prev_distance = float(tokens[1]) - float(samples_lines[i-1].split()[1]) 
      next_distance = float(samples_lines[i+1].split()[1]) - float(tokens[1]) 
    # this is a combination of 33 samples, 450k chip indicator
    # and distances to previous neighbour and next neighbour
    samples_distances = samples+[float(tokens[37])]+[prev_distance]+[next_distance]
    samples_distances_lines.append(samples_distances)

  for i in range(len(samples_distances_lines)):
    # check if the segment has nan in 450k chip
    if 'nan' not in truth_lines[i].split():
      samples_distances = samples_distances_lines[i]
      if i == 0:
        prev_samples_distances = [float(0)]*sample_size
        next_samples_distances = samples_distances_lines[i+1]
      elif i == len(samples_lines)-1:
        prev_samples_distances = samples_distances_lines[i-1]
        next_samples_distances = [float(0)]*sample_size
      else:
        prev_samples_distances = samples_distances_lines[i-1]
        next_samples_distances = samples_distances_lines[i+1]

      combined_samples_distances = samples_distances[:sample_size] + prev_samples_distances[:sample_size] + next_samples_distances[:sample_size] + samples_distances[sample_size+1:]
      #print samples_distances[:sample_size], len(samples_distances[:sample_size])
      #print prev_samples_distances[:sample_size], len(prev_samples_distances[:sample_size])
      #print next_samples_distances[:sample_size], len(next_samples_distances[:sample_size])
      #print combined_samples_distances, len(combined_samples_distances)

      if samples_distances[sample_size] == float(1):
        samples_train.append(combined_samples_distances)
      else:
        samples_test.append(combined_samples_distances)
    else:
      pass
      #print line
      #print truth_lines[samples_lines.index(line)]

  print len(samples_train),"trains", len(samples_test), "tests"

  matrix_width = 3*sample_size+2
  matrix_train  = numpy.zeros(shape=(len(samples_train),matrix_width), dtype=numpy.uint8)
  print matrix_train.shape
  for i in range(len(samples_train)):
    for j in range(matrix_width):
      matrix_train[i,j] = samples_train[i][j]

  matrix_test = numpy.zeros(shape=(len(samples_test),matrix_width), dtype=numpy.uint8)
  print matrix_test.shape
  for i in range(len(samples_test)):
    for j in range(matrix_width):
      matrix_test[i,j] = samples_test[i][j]
  
  matrix_train.tofile("../data/X%s_chr%d_cutoff_%d_len_%d.dat" %(nan_code,chr_no,cutoff,len(samples_train)))
  matrix_test.tofile("../data/X_test%s_chr%d_cutoff_%d_len_%d.dat" %(nan_code,chr_no,cutoff,len(samples_test)))


  for line in truth_lines:
    #print line
    tokens = line.split()
    #print tokens, len(tokens)
    if 'nan' not in tokens:
      truth = process_nan(tokens[4:5],nan_code)
      #print truth, len(truth)
      if tokens[5] == '1':
        truth_train.append(truth)
      else:
        truth_test.append(truth)

  print len(truth_train),"trains", len(truth_test), "tests"
      
  vector_train  = numpy.zeros(shape=(len(truth_train),1), dtype=numpy.uint8)
  print vector_train.shape
  for i in range(len(truth_train)):
    for j in range(1):
      vector_train[i,j] = truth_train[i][j]

  vector_test = numpy.zeros(shape=(len(truth_test),1), dtype=numpy.uint8)
  print vector_test.shape
  for i in range(len(truth_test)):
    for j in range(1):
      vector_test[i,j] = truth_test[i][j]
      
  vector_train.tofile("../data/y%s_chr%d_cutoff_%d_len_%d.dat" %(nan_code,chr_no,cutoff,len(truth_train)))
  vector_test.tofile("../data/y_test%s_chr%d_cutoff_%d_len_%d.dat" %(nan_code,chr_no,cutoff,len(truth_test)))


generate_dat(chr_no, cutoff, '0')
generate_dat(chr_no, cutoff, 'A')
generate_dat(chr_no, cutoff, 'R') 

