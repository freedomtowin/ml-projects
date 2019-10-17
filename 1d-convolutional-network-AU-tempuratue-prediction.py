#!/usr/bin/env python
# coding: utf-8

"""
This script implement a 1-d convolutional neural network model in tensorflow to predict AU temperatures
"""


import pandas as pd
import time
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import datetime
from scipy import signal
import tensorflow as tf
import os

df=pd.read_csv("data/daily-minimum-temperatures-and-rain-fall.csv")
df['Date']=[datetime.datetime.strptime(t[0],"%m/%d/%y") for t in df[['Date']].values]
df.sort_values(by='Date',inplace=True)


def replace_with_nan(x):
    if '?' in x:
        return np.nan
    else:
        return float(x)
df.Temp = df.Temp.apply(lambda x: replace_with_nan(str(x)))
df.Rain = df.Rain.apply(lambda x: replace_with_nan(str(x)))


def interpolate_nans(y):
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    nans, x = nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y

df.loc[:,'Temp'] = interpolate_nans(df['Temp'])
df.loc[:,'Rain'] = interpolate_nans(df['Rain'])

train_percentage = 0.8

y = df[['Temp']].values[:int(len(df)*train_percentage)]
plt.figure(figsize=(10,5))
plt.plot(df[['Date']].values,df[['Temp']].values)
plt.title('temperature by year')
plt.show()

def autocorrelation(x):
    result = np.correlate(x, x, mode='same')
    result = result[result.size//2:]
    return result

def derivative(x):
    x = np.convolve(x,[-1,0,1],mode='full')
    x = x[1:-1]
    return x

def integral(x,w=50):
    x = np.convolve(x,np.ones(w)/w,mode='full')
    x = x[w//2:-w//2+1]
    return x


# The autocorrelation function is used on the target
# The results of this function are detrended
# The first value is discarded as it is lag 0

ac = autocorrelation(y.ravel())
ac = signal.detrend(ac)
t = np.arange(0,ac.size)

ac = ac[1:]
t = t[1:]

# The derivative (difference) of the integral (smooth) of gives to locations of
# the peaks
# This can be used instead of the next method
d = derivative(integral(ac))
peaks = np.where(np.abs(d)<8)[0]

#Find the large peaks
strong_corr = np.where(np.abs(ac)>np.mean(ac)+np.std(ac))[0]
plt.figure(figsize=(10,5))
plt.plot(t[1:],ac[1:])
plt.plot(t[strong_corr],ac[strong_corr],'g.')
plt.plot(t[peaks],ac[peaks],'r.')
plt.title('autocorrelation')
plt.show()
print('autocorrelation')
print(t[peaks])

#save the most correlated lags
autocorrelation_peaks = t[strong_corr]
# Subset the correlated lags to those less than 400
indices = autocorrelation_peaks.astype(np.int)
indices = indices[indices<400]
# The mask will be used later to select the lags
mask = np.zeros(400).astype(np.bool) #np.ones_like(a,dtype=bool)
mask[indices] = True

lag = 400

def create_permutation(size):
    return np.random.permutation(size)

def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)

    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def create_train_test(data,percentage,permutation):
    c = int(data.shape[0] * percentage)
    data_train = data[0:c]
    shuffled_train = np.empty(data_train.shape)
    for old_index, new_index in enumerate(permutation):
        shuffled_train[new_index] = data_train[old_index]
    data_test = data[c:]
    return shuffled_train,data_test


def split_into_chunks(data,lag):
    result = np.zeros((data.shape[0]-lag,lag,data.shape[1]))
    for i in range(lag,data.shape[0]):
        result[i-lag] = data[i-lag:i]
    return result


# Create shuffled data for training and validation dataset
y = df[['Temp']].values[lag:]

X = split_into_chunks(df[['Temp','Rain']].values,lag)[:,:,:]



perm = create_permutation(int(y.size * 0.8))

y_train, y_test = create_train_test(y,percentage=0.8,permutation=perm)

X_train, X_test = create_train_test(X,percentage=0.8,permutation=perm)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


#create 1-d convolution network
num_series = 2
input_X = tf.placeholder(tf.float32,[None,lag,num_series],'input_X')

input_y = tf.placeholder(tf.float32,[None,1],'input_y')


conv0 = tf.layers.conv1d(inputs=input_X,filters=18,kernel_size=400,strides=25,padding='same',activation='relu')


weights_1 = tf.Variable(tf.random_normal(((conv0.shape[1]*conv0.shape[2]).value,1),
                    mean=0.0,stddev=0.01,dtype='float32'),'weights1')

b1 = tf.Variable(tf.zeros((1,1),dtype='float32'))

flat = tf.reshape(conv0,[-1,conv0.shape[1]*conv0.shape[2]])

predicted_y = tf.add(tf.matmul(flat,weights_1),b1)

error = tf.reduce_mean(tf.square(predicted_y-input_y))


optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(error)


#train network
s.run(tf.global_variables_initializer())
batchsize=300
for epoch in range(0,300):
    
    rand_sample = np.random.choice(X_train.shape[0],size=batchsize,replace=False)
    X_batch = X_train[rand_sample]
    y_batch = y_train[rand_sample]
    
    s.run(optimizer,feed_dict={input_X:X_batch[:],input_y:y_batch[:]})
    if epoch%10==0:
        print(s.run(error,feed_dict={input_X:X[:],input_y:y[:]}))


#Plot predicted vs actual
plt.figure(figsize=(10,5))
plt.plot(y,alpha=0.5)
plt.plot(s.run(predicted_y,feed_dict={input_X:X[:],input_y:y[:]}),alpha=0.5)
plt.title




