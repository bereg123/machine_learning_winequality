import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import csv
import random
import math
import operator
#This code is adapted from https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
#The loadDataset method loads the data from the csv file and split it into the train test sets based on the split input
#the split should be between 0-1, which means  split*100% data will be split into the training set.

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
        #this make sure we get the same train_set data each time we run our code
        np.random.seed(30)

	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile,delimiter= ";")
	    header = next(lines)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])


#The euclideanDistance method calculate the distance between 2 data sets using euclideanDistance

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(distance)

#The getNeighbors method returns the k nearest neighbours based on the distance caculated by euclideanDistance method
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
	
#The getResponse method returns the average quality from the k nearest  neighbours
def getResponse(neighbors):
    length = len(neighbors)
    quality = []
    for x in range(0,length):
        c=neighbors[x][-1]
        quality.append(float(c))
    result=np.mean(quality)
    return result
    
#The getTestError method calculated the mean square error for the test set and returns the result
def getTestError(testSet, getResponse):
    error = len(getResponse)
    err=[]
    meanqrerr =[]
    for x in range(0,len(getResponse)):
        error=(float(testSet[x][-1]) - getResponse[x] )**2
        meanqrerr.append(error)
        ems = np.sqrt(error)
        error_rate=ems/float(testSet[x][-1])
        err.append(error_rate)
    err = np.mean(err)
    total = np.sum(meanqrerr)
    total/= len(getResponse)
    return total
	
#The getTrainingError method calculated the mean square error for the training set and returns the result     
def getTrainingError(trainingSet, getResponse):
    error = len(getResponse)
    err=[]
    meanqrerr =[]
    for x in range(0,len(getResponse)):
        error=(float(trainingSet[x][-1]) - getResponse[x] )**2
        meanqrerr.append(error)
        ems = np.sqrt(error)
        error_rate=ems/float(trainingSet[x][-1])
        err.append(error_rate)
    err = np.mean(err)
    total = np.sum(meanqrerr)
    total/= len(getResponse)
    return total

#The test error method print out the final result for test error and plot it
def testError():
	# prepare data
	np.random.seed(30)
	trainingSet=[]
	testSet=[]
	split = 0.75
	loadDataset('../winequality-red.csv', split, trainingSet, testSet)
	#print 'Train set: ' + repr(len(trainingSet))
	#print 'Test set: ' + repr(len(testSet))
	# generate predictions 
	Kvalue=[]
	Avalue=[]
	for k in range(1,20):
	  Kvalue.append(k)  
	  predictions=[]
          for x in range(len(testSet)):
		   neighbors = getNeighbors(trainingSet, testSet[x], k)
		   result = getResponse(neighbors)
		   predictions.append(result)
		   #print(len(testSet))
		   #print(len(predictions))
		   #print predictions[x]
		   #print testSet[x][-1]
		  # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	  erorrate = getTestError(testSet, predictions)
	  print( '> test error='+erorrate)
	  	
	  Avalue.append(float(erorrate))
         
		
	#print(Kvalue)
	#print(Avalue)
	fig1 = plt.figure()
	fig1.suptitle("test error against k ")
	ax1=fig1.add_subplot(1,1,1)
	ax1.set_xlabel("k")
	ax1.set_ylabel("test error")
	ax1.plot(Kvalue,Avalue,'-',markersize=1)
	fig1.show
	plt.show()
	
def trainError():
	# prepare data
	np.random.seed(30)
	trainingSet=[]
	testSet=[]
	split = 0.75
	loadDataset('../winequality-red.csv', split, trainingSet, testSet)
	#print 'Train set: ' + repr(len(trainingSet))
	#print 'Test set: ' + repr(len(testSet))
	# generate predictions 
	Kvalue=[]
	Avalue=[]
	for k in range(1,20):
	  Kvalue.append(k)  
	  predictions=[]
          for x in range(len(testSet)):
		   neighbors = getNeighbors(trainingSet, testSet[x], k)
		   result = getResponse(neighbors)
		   predictions.append(result)
		   #print(len(testSet))
		   #print(len(predictions))
		   #print predictions[x]
		   #print testSet[x][-1]
		  # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	  erorrate = getTestError(trainingSet, predictions)
	  print( '> test error='+erorrate)
	  	
	  Avalue.append(float(erorrate))
         
		
	#print(Kvalue)
	#print(Avalue)
	fig1 = plt.figure()
	fig1.suptitle("test error against k ")
	ax1=fig1.add_subplot(1,1,1)
	ax1.set_xlabel("k")
	ax1.set_ylabel("test error")
	ax1.plot(Kvalue,Avalue,'-',markersize=1)
	fig1.show
	plt.show()	
testError
trainError()