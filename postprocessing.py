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
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


newTrain = pd.read_csv('newTrain.csv')


# In[3]:


newTest = pd.read_csv('newTest.csv')


# In[4]:


newTrain.drop(newTrain.columns[0],axis=1,inplace=True)
newTest.drop(newTest.columns[0],axis=1,inplace=True)


# In[5]:


newTrain['truth'] += 1
newTrain['truth'] /= 2

newTrain['characters2'] += 1
newTrain['characters2'] /= 2

newTrain['characters3'] += 1
newTrain['characters3'] /= 2

newTrain['characters4'] += 1
newTrain['characters4'] /= 2

newTrain['combined'] += 1
newTrain['combined'] /= 2

newTrain['words'] += 1
newTrain['words'] /= 2



newTest['truth'] += 1
newTest['truth'] /= 2

newTest['characters2'] += 1
newTest['characters2'] /= 2

newTest['characters3'] += 1
newTest['characters3'] /= 2

newTest['characters4'] += 1
newTest['characters4'] /= 2

newTest['combined'] += 1
newTest['combined'] /= 2

newTest['words'] += 1
newTest['words'] /= 2


# In[6]:


newTrain.head()


# In[7]:


newTrain.shape


# In[8]:


newTest.shape


# In[9]:


train_x = newTrain.drop(columns=['truth','created','name','text'])
train_x['day'] = train_x['day'].astype(int)
train_x['hr'] = train_x['hr'].astype(int)
train_x['month'] = train_x['month'].astype(int)

train_y = newTrain[['truth']]


# In[10]:


test_x = newTest.drop(columns=['truth','created','name','text'])
test_x['day'] = test_x['day'].astype(int)
test_x['hr'] = test_x['hr'].astype(int)
test_x['month'] = test_x['month'].astype(int)

test_y = newTest[['truth']]


# In[11]:


train_x.describe(include='all')


# In[12]:


# sns.pairplot(train_x)


# In[13]:


# sns.heatmap(train_x.corr(), annot=True)


# In[14]:


train_x.head()


# In[15]:


train_y.head()


# In[16]:


train_x_dropped = train_x.drop(columns=['day','hr','month','capsInText','hashTags','mentions'])


# In[17]:


test_x_dropped = test_x.drop(columns=['day','hr','month','capsInText','hashTags','mentions'])


# In[18]:


train_x_dropped.head()


# In[19]:


# np_train_x = train_x.to_numpy()
np_train_y = train_y.to_numpy()

# np_test_x = test_x.to_numpy()
np_test_y = test_y.to_numpy()

np_train_x = train_x_dropped.to_numpy()
np_test_x = test_x_dropped.to_numpy()


# In[20]:


np_train_x


# In[21]:


np_train_y


# In[22]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
np_train_x = sc.fit_transform(np_train_x)
np_test_x = sc.fit_transform(np_test_x)


# In[23]:


np_train_x


# In[24]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.callbacks import TensorBoard
from sklearn import preprocessing
from keras.regularizers import l2, l1
from keras.utils import plot_model


# In[25]:


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# In[26]:


classifier = Sequential()
# classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=7))
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=7))
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))


# In[27]:


classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])


# In[28]:


history = classifier.fit(np_train_x, np_train_y, batch_size=100, epochs=10, validation_split=0.1)


# In[29]:


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[30]:


classifier.evaluate(np_train_x, np_train_y, verbose=0)


# In[31]:


y_pred=classifier.predict(np_test_x)
y_pred =(y_pred>0.5)


# In[32]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np_test_y, y_pred)
print(cm)


# In[ ]:





# In[35]:


# from sklearn.model_selection import RandomizedSearchCV

# classifier = Sequential()
# distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
# clf = RandomizedSearchCV(classifier, distributions, random_state=0)
# search = clf.fit(np_train_x, np_train_y)
# search.best_params_


# In[36]:


# from sklearn.model_selection import GridSearchCV

# classifier = Sequential()
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# clf = GridSearchCV(classifier, parameters)
# search = clf.fit(np_train_x, np_train_y)
# search.best_params_


# In[33]:


from sklearn.ensemble import RandomForestRegressor


# In[34]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# In[35]:


rf.fit(np_train_x, np_train_y)


# In[36]:


y_pred_rf = rf.predict(np_test_x)


# In[37]:


y_pred_rf =(y_pred_rf>0.5)


# In[38]:


cm = confusion_matrix(np_test_y, y_pred_rf)
print(cm)


# In[39]:


feature_list = list(train_x_dropped.columns)


# In[40]:


# Visualization
from sklearn.tree import export_graphviz
import pydot
tree = rf.estimators_[5]
from sklearn.tree import export_graphviz
import pydot# Pull out one tree from the forest
tree = rf.estimators_[5]
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# graph.write_png('tree.png')


# In[41]:


# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(np_train_x, np_train_y)
tree_small = rf_small.estimators_[5]
export_graphviz(tree_small, out_file='small_tree.dot', feature_names=feature_list, rounded=True, precision=1)
# (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
# graph.write_png('small_tree.png');


# In[42]:


y_pred_rf_small = rf_small.predict(np_test_x)
y_pred_rf_small =(y_pred_rf_small>0.5)
cm = confusion_matrix(np_test_y, y_pred_rf_small)
print(cm)


# In[ ]:




