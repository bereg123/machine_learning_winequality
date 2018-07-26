import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import csv
import random

#This code is adpated from https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

#The loadDataset loads the data from the csv file and split data into trainning 
# and test set based on the split input. split should be in the range 0f  0-1, 
#split*100 means the percentage of data that got splited into the training set.
#

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
        np.random.seed(30)
	with open(filename, 'rb') as csvfile:
	    np.random.seed(30)
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
import math
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(distance)


#The getNeighbors method returns the k nearest neighbours based on the distance caculated by euclideanDistance method	
import operator 
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

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] is predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

#The test error method print out the final result for test error and plot it
def testError():
	# prepare data
	np.random.seed(30)
	trainingSet=[]
	testSet=[]
	newSet=[]
	split = 0.75
	loadDataset('../winequality-red.csv', split, trainingSet, testSet)
	#print 'Train set: ' + repr(len(trainingSet))
	#print 'Test set: ' + repr(len(testSet))
	# generate predictions
	Kvalue=[]
	Avalue=[]
	Tvalue=[]
	for k in range(1,40):
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
		   print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	  erorrate = getTestError(testSet, predictions)
	  trainerorrate = getTestError(trainingSet, predictions)
	  print(erorrate)
	  #print( trainerorrate)
	  	
	  Avalue.append(float(erorrate))
	  #Tvalue.append(float(trainerorrate))
	 
          #print('Accuracy: ' + repr(accuracy) + '%')
		
	#print(Kvalue)
	#print(Avalue)
	#print(Tvalue)
	
	fig1 = plt.figure()
	fig1.suptitle("test error MSE against k ")
	ax1=fig1.add_subplot(1,1,1)
	ax1.set_xlabel("k")
	ax1.set_ylabel("test error MSE")
	ax1.plot(Kvalue,Avalue,'-',markersize=1)
	fig1.show
	plt.show()
	optimal_k=Kvalue[Avalue.index(min(Avalue))]
	print(optimal_k)
	


def crossValidation ():
    np.random.seed(30)
    folds=5
    trainingSet=[]
    testSet=[]
    split = 0.75
    Kvalue=[]
    Cross_Validation=[]
    loadDataset('../winequality-red.csv', split, trainingSet, testSet)
    files = open('../winequality-red.csv','r')
    lines = csv.reader(files,delimiter= ";")
    #lines = pd.read_csv('../winequality-red.csv')
    #header = next(lines)
    #dataSet = list(lines)
    header = next(lines)
        # print("\n\nImporting data with fields:\n\t" + ",".join(header))

        # creating an empty list to store each row of data
    data = []

    for row in lines:
            # for each row of data 
            # converting each element (from string) to float type
            row_of_floats = list(map(float, row))

            # now storing in our data list
            data.append(row_of_floats)

        # print("There are %d entries." % len(data))

        # converting the data (list object) into a numpy array
    data_as_array = np.array(data)
   
    
    length = round(len(data_as_array)/folds)
    average = [] 
    for k in range(1,40):
        Kvalue.append(k)   
        for y in range(1,folds):
            testSet = data_as_array[int(y*length):int((y+1)*length)]
            trainingSet = np.delete(data_as_array,[i for i in range(int(y*length),int((y+1)*length))],0)
            error = []
            pred = []
            for x in range (1,50):
                neighbors = getNeighbors(trainingSet, testSet[x], k)
                result = getResponse(neighbors)
                pred.append(result)
            errot_rate = getTestError(testSet,pred)
            error.append(errot_rate)
        average.append(error)
        Cross_Validation.append(np.mean(error))
        print(np.mean(error))
    fig1 = plt.figure()
    fig1.suptitle("Cross validation test error MSE against k ")
    ax1=fig1.add_subplot(1,1,1)
    ax1.set_xlabel("k")
    ax1.set_ylabel("test error MSE")
    ax1.plot(Kvalue,Cross_Validation,'-',markersize=1)
    fig1.show
    plt.show()
    optimal_k=Kvalue[error.index(min(Cross_Validation))]
    print(optimal_k)


	

testError()
#crossValidation()

#testErrornew()

    