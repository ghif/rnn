import numpy as np
import random

from sklearn import preprocessing
from keras.utils import np_utils

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# def sample(f, x, n):
#     # f: theano function for activating the second last layer values
#     # x: numpy 1D array associated with a character
#     # n : number of sampled characters
#     [_, _, vocab_size]= x.shape
#     ixes = []

#     for t in range(n):
#         y = f(x)
#         exp_y = np.exp(y)
#         p = exp_y/np.sum(exp_y)

#         ix = np.random.choice(range(vocab_size), p=p.ravel())
#         x = np.zeros((vocab_size, 1))
#         x[ix] = 1
#         x = x.reshape((1,1,vocab_size)).astype('float32')
#         ixes.append(ix)

#     return ixes

def create_train_val(X, Y, p=0.7):
    [n, t, d] = X.shape
    n_train = int(p*n)
    n_val = n - n_train


    inds = np.random.permutation(n)

    
    inds_train = inds[:n_train]
    inds_val = [i for i in inds if i not in set(inds_train)]
    X_train = X[inds_train,:,:].astype('float32')
    Y_train = Y[inds_train,:,:].astype('float32')

    X_val = X[inds_val,:,:].astype('float32')
    Y_val = Y[inds_val,:,:].astype('float32')
    return (X_train, Y_train, X_val, Y_val)


def sample(z, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(z) / temperature
    a = np.exp(z) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

