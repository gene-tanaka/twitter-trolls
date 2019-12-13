#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import collections
import math
import sys
import copy
import pandas as pd
import os
import datetime
import numpy as np


# In[2]:


fake = pd.read_csv(os.getcwd()+'/fake.csv')
fake = fake[['user_key','text','created_str']]


# In[3]:


real = pd.read_csv(os.getcwd()+'/real.csv')
real = real[['created','name','text']]
real['truth'] = 1
fake['truth'] = -1

fake = fake.rename(columns={"created_str": "created", "user_key": "name", })
real['created'] = pd.to_datetime(real['created'])
fake['created'] = pd.to_datetime(fake['created'])

real['month'] = real['created'].dt.month
fake['month'] = fake['created'].dt.month
real['day'] = real['created'].dt.day
fake['day'] = fake['created'].dt.day
real['hr'] = real['created'].dt.hour
fake['hr'] = fake['created'].dt.hour


# In[4]:


real.head()


# In[5]:


fake.head()


# In[6]:


real.shape


# In[7]:


fake.shape


# In[ ]:


train_real = real.head(10000)
train_fake = fake.head(10000)
train = train_real.append(train_fake, sort=True)
train = train.sample(frac=1).reset_index(drop=True)

test_real = real.tail(1000)
test_fake = fake.tail(1000)
test = test_real.append(test_fake, sort=True)
test = test.sample(frac=1).reset_index(drop=True)

trainExamples = []

for index, row in train.iterrows():
    trainExamples.append((row['text'], row['truth']))
trainExamples = tuple(trainExamples)

testExamples = []

for index, row in test.iterrows():
    testExamples.append((row['text'], row['truth']))
testExamples = tuple(testExamples)


# In[9]:


# trainExamples


# In[10]:


def extractWordFeatures(x):
    features = {}
    for word in x.split():
        if word not in features:
            features[word] = 0
        features[word] += 1
    return features


# In[11]:


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    def hingeLoss(w, i):
        x, y = trainExamples[i]
        features = featureExtractor(x)
        return max(1 - dotProduct(weights,features)*y, 0)
        
    def dHingeLoss(w, i):
        x, y = trainExamples[i]
        features = featureExtractor(x)
        if dotProduct(weights,features)*y < 1:
            features.update((a, b*-y) for a, b in features.items())
            return features
        else:
            features.update((a, 0) for a, b in features.items())
            return features
        
    weights = {}

    n = len(trainExamples)
    for t in range(numIters):
        for i in range(n):
            gradient = dHingeLoss(weights, i)
            increment(weights, -eta, gradient)
    predictor = lambda x : 1 if dotProduct(featureExtractor(x), weights) > 0 else -1
    print('Train Error: {}'.format(evaluatePredictor(trainExamples, predictor)))
    print('Test Error: {}'.format(evaluatePredictor(testExamples, predictor)))
    trainPredictions = predictions(trainExamples, predictor)
    testPredictions = predictions(testExamples, predictor)
    return weights, trainPredictions, testPredictions


# In[12]:


def combinedFeatures(x):
    features = {}
    for word in x.split():
        if word not in features:
            features[word] = 0
        features[word] += 1
    
    n = 2
    x = x.replace(" ","")
    for i in range(len(x) - n + 1):
        if x[i:i+n] not in features:
            features[x[i:i+n]] = 0
        features[x[i:i+n]] += 1
    return features


# In[13]:


def extractCharacterFeatures(n):
    def extract(x):
        x = x.replace(" ","")
        nGrams = {}
        for i in range(len(x) - n + 1):
            if x[i:i+n] not in nGrams:
                nGrams[x[i:i+n]] = 0
            nGrams[x[i:i+n]] += 1
        return nGrams
    return extract


# In[14]:


def dotProduct(d1, d2):
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in list(d2.items()))
    
def increment(d1, scale, d2):
    for f, v in list(d2.items()):
        d1[f] = d1.get(f, 0) + v * scale
        
def evaluatePredictor(examples, predictor):
    error = 0
    for x, y in examples:
        if predictor(x) != y:
            error += 1
    return 1.0 * error / len(examples)

def predictions(examples, predictor):
    predictions = []
    for x, y in examples:
        predictions.append(predictor(x))
    return predictions


# In[15]:


featureExtractor = combinedFeatures
weightsCombined, trainPredictionsCombined, testPredictionsCombined = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)


# In[16]:


featureExtractor = extractWordFeatures
weightsWords, trainPredictionsWords, testPredictionsWords = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)


# In[17]:


featureExtractor = extractCharacterFeatures(3)
weightsCharacters3, trainPredictionsCharacters3, testPredictionsCharacters3 = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)


# In[18]:


featureExtractor = extractCharacterFeatures(4)
weightsCharacters4, trainPredictionsCharacters4, testPredictionsCharacters4 = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)


# In[19]:


featureExtractor = extractCharacterFeatures(2)
weightsCharacters2, trainPredictionsCharacters2, testPredictionsCharacters2 = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)


# In[22]:


def addFeatures(df, dataType):
    if dataType == 'train':
        df['characters2'] = trainPredictionsCharacters2
        df['characters4'] = trainPredictionsCharacters4
        df['characters3'] = trainPredictionsCharacters3
        df['combined'] = trainPredictionsCombined
        df['words'] = trainPredictionsWords
    else:
        df['characters2'] = testPredictionsCharacters2
        df['characters4'] = testPredictionsCharacters4
        df['characters3'] = testPredictionsCharacters3
        df['combined'] = testPredictionsCombined
        df['words'] = testPredictionsWords
    df = addNumbersInName(df)
    df = addSpecialCharactersInText(df)
    df = addCapsInText(df)
    return df


# In[23]:


def addNumbersInName(df):
    numbersInName = []
    for index, row in df.iterrows():
        count = 0
        text = row['name']
        for i in range(10):
            count += text.count(str(i))
        numbersInName.append(count)
    df['numbersInName'] = numbersInName
    return df


# In[40]:


def addSpecialCharactersInText(df):
    specialCharsText = []
    for index, row in df.iterrows():
        count = 0
        text = row['text']
        for ch in text:
            if not ((ch >= 'A' and ch <= 'Z') or (ch >= 'a' and ch <= 'z')):
                count += 1
        specialCharsText.append(1.0*count/len(text))
    df['specialCharsText'] = specialCharsText
    return df


# In[41]:


def addCapsInText(df):
    capsInText = []
    for index, row in df.iterrows():
        count = 0
        text = row['text']
        for ch in text:
            if ch >= 'A' and ch <= 'Z':
                count += 1
        capsInText.append(1.0*count/len(text))
    df['capsInText'] = capsInText
    return df


# In[42]:


newTrain = addFeatures(train, 'train')
newTest = addFeatures(test, 'test')


# In[43]:


newTrain.to_csv('newTrain.csv')
newTest.to_csv('newTest.csv')


# In[47]:


newTrain.head()


# In[45]:


train_x = newTrain.drop(columns=['truth','created','name','text'])


# In[46]:


train_x['day'] = train_x['day'].astype(int)
train_x['hr'] = train_x['hr'].astype(int)
train_x['month'] = train_x['month'].astype(int)


# In[31]:


train_x.dtypes


# In[32]:


train_y = newTrain[['truth']]


# In[33]:


train_x.head()


# In[34]:


train_y.head()


# In[ ]:




