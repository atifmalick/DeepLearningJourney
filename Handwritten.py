#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


(x_train,y_train), (x_test, y_test)= keras.datasets.mnist.load_data()


# In[3]:


len(x_train)


# In[4]:


len(x_test)


# In[5]:


x_train[0].shape


# In[6]:


x_train[0]


# In[7]:


plt.matshow(x_train[1])


# In[8]:


y_train[2]


# In[9]:


x_train.shape


# In[10]:


x_train_flattended=x_train.reshape(len(x_train),28*28) #convert 2d to flatten shape
x_test_flattended=x_test.reshape(len(x_test),28*28)


# In[11]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid') #Dense means each neuron are connected with other neuron
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattended, y_train, epochs=5)


# In[12]:


#Scale the dataset to improve accuracy
x_train= x_train/255
x_test=x_test/255


# In[13]:


x_train[0]


# In[14]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid') #Dense means each neuron are connected with other neuron
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattended, y_train, epochs=5)


# In[15]:


model.evaluate(x_test_flattended, y_test)


# In[16]:


y_predicted = model.predict(x_test_flattended)
y_predicted[0]


# In[17]:


plt.matshow(x_test[0])


# In[18]:


#np.argmax finds a maximum element from an array and returns the index of it
np.argmax(y_predicted[0])


# In[19]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]


# In[20]:


y_predicted_labels[:5]


# In[21]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[23]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicate')
plt.ylabel('truth')


# In[27]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattended, y_train, epochs=5)


# In[28]:


model.evaluate(x_test_flattended,y_test)


# In[29]:


y_predicted = model.predict(x_test_flattended)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[30]:


#Using Flatten layer so that we don't have to call .reshape on input dataset
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)


# In[31]:


model.evaluate(x_test,y_test)


# In[ ]:




