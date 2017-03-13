
# coding: utf-8

# In[21]:

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csc_matrix
from sklearn import svm
from __future__ import division


# In[22]:

# function to read in the sparsematrix format used by CS229
def readMatrix(filenname):
    
    file_matrix = open(filenname, "r")
    lines = file_matrix.readlines()
    
    rowscols = (int(lines[1].split()[0]), int(lines[1].split()[1]))
    tokenlist = lines[2]
    matrix = lil_matrix((rowscols[1],rowscols[0]), dtype=np.int8)
    category = np.zeros(rowscols[0])
    line = np.zeros(rowscols[1])
    
    for i in range(rowscols[0]):
        line = [int(lines[3+i].split()[j]) for j in range(len(lines[3+i].split()))]
        category[i]=int(line[0])
        cumsum = np.cumsum(line[1::2])
        for j in range(len(line[1::2])-1):
            matrix[cumsum[j],i] = line[2::2][j]
    matrix.tocsc();
    matrix = matrix.transpose()
        
    return (tokenlist, category, matrix)


# In[23]:

# Trains the Naive Bayes algorithm given a category vector and a desgin matrix.
# Returns the conditional probabilities logSpamPhi, logNoSpamPhi and the prioris logSpamPrior, logNoSpamPrior
# The explicit form of these probabilities can be deduced by maximizing the joint log likelihood of the problem. 
def trainNaiveBayes(category, matrix):
    
    train = matrix.toarray()
    documents = train.shape[0]
    words = train.shape[1]
    
    spamTrain = np.array( [ train[i][:]  for i in range(documents) if category[i] == 1 ])
    noSpamTrain = np.array([train[i][:] for i in range(documents) if category[i] == 0 ])
    spamDocuments = spamTrain.shape[0]
    noSpamDocuments = noSpamTrain.shape[0]
    
    logSpamPrior = np.log(spamDocuments / documents)
    logNoSpamPrior = np.log(noSpamDocuments / documents)
    allSpamWords = sum([sum([spamTrain[l][k] for l in range(spamDocuments)]) for k in range(words)])
    allNoSpamWords = sum([sum([noSpamTrain[l][k] for l in range(noSpamDocuments)]) for k in range(words)])  
    
    logSpamPhi = np.log([( 1 + sum([spamTrain[i][j] for i in range(spamDocuments)])) /(words +allSpamWords)
                  for j in range(words)])
    
    logNoSpamPhi = np.log([ (1 + sum([noSpamTrain[i][j] for i in range(noSpamDocuments)])) / (words + allNoSpamWords)
                    for j in range(words)])
    
    return (logSpamPrior, logNoSpamPrior, logSpamPhi, logNoSpamPhi)


# In[24]:

# Uses Bayes' theorem to predict the class labels from the conditional probabilities logSpamPhi, logNoSpamPhi
# and the prioris logSpamPrior, logNoSpamPrior

def isSpam(mail, logSpamPrior, logNoSpamPrior, logSpamPhi, logNoSpamPhi):
    
    logPosterioriProbSpam = sum([mail[j]*logSpamPhi[j] for j in range(mail.shape[0])]) + logSpamPrior
    logPosterioriProbNoSpam = sum([mail[j]*logNoSpamPhi[j] for j in range(mail.shape[0])]) + logNoSpamPrior
    
    return int(logPosterioriProbSpam > logPosterioriProbNoSpam)


# In[25]:

# sort the tokens in order of how indicative there are for spam
def sortTokens(tokens, logSpamPrior, logNoSpamPrior, logSpamPhi, logNoSpamPhi):
    sortedTokens = sorted(tokens.split(), key =lambda(token): np.log( ( logSpamPhi[tokens.split().index(token)] + logSpamPrior) 
                                                             /(logNoSpamPhi[tokens.split().index(token)] + logNoSpamPrior)))
    
    return sortedTokens


# In[26]:

def trainSVM(category, matrix):
    svmCategory = [2*category[i]-1 for i in range(len(category))]
    classifier = svm.LinearSVC()
    classifier.fit(matrix,svmCategory)
    
    return classifier
    


# In[48]:

# Finally print the errors. 
def printSolution():
    
    trainMatrix = ["MATRIX.TRAIN.50","MATRIX.TRAIN.100", "MATRIX.TRAIN.200", "MATRIX.TRAIN.400",
                   "MATRIX.TRAIN.800", "MATRIX.TRAIN.1400","MATRIX.TRAIN"]
    (testlist, categoryTest, test) = readMatrix("MATRIX.TEST")
    testArray = test.toarray()
    
    for matrix in trainMatrix:
        
        (tokenlist, category, trainMatrix) = readMatrix(matrix)
        (logSpamPrior, logNoSpamPrior, logSpamPhi, logNoSpamPhi)= trainNaiveBayes(category, trainMatrix)
        classifer = trainSVM(category, trainMatrix)
        correct = sum([1 for i in range(testArray.shape[0]) 
                if (isSpam(testArray[i][:], logSpamPrior, logNoSpamPrior, logSpamPhi, logNoSpamPhi) == (categoryTest[i]==1))])/ testArray.shape[0]
        correctSVM = sum([1 for i in range(testArray.shape[0]) 
            if ( (classifer.predict(testArray[i][:].reshape(1,-1))==1) == (categoryTest[i]==1))])/ testArray.shape[0]
        
        print  "The error using Naive Bayes and the training sample " + matrix + " is: " + str((1-correct)*100) +"%."
        print  "The error using an SVM with a linear kernel and the training sample " + matrix + " is: " + str((1-correctSVM)*100) +"%."

    sortedTokens = sortTokens(tokenlist,logSpamPrior, logNoSpamPrior, logSpamPhi, logNoSpamPhi)
    print "The five most indicative tokens for spam are:"
    for i in range(5):
        print sortedTokens[i]


# In[ ]:




# In[49]:

printSolution()


# In[ ]:




# In[ ]:




# In[ ]:



