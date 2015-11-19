from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2


import numpy as np
import sys

from myutils import *
import cPickle as pickle
import gzip


seqlen = 50 # 
lettersize = 40
clipval = 5
learning_rate = 6e-3

# Data I/O
vocabs = initvocab('data/samples.txt', seqlen)
text = vocabs['text']
sents = vocabs['sents']
vocab = vocabs['vocab']
char_indices = vocabs['char_indices']
indices_char = vocabs['indices_char']

inputsize = len(vocab)
outputsize = inputsize
n = len(sents)


print('Build GRU...')


model = Sequential()
model.add(Embedding(inputsize, lettersize))

model.add(GRU(76, 
    return_sequences=True, 
    inner_activation='sigmoid',
    activation='tanh',
    truncate_gradient=clipval,
    input_dim=inputsize)
)
# model.add(Dropout(0.2))
model.add(GRU(80, 
    return_sequences=True,
    inner_activation='sigmoid',
    activation='tanh',
    truncate_gradient=clipval
    )
)
# # model.add(Dropout(0.2))
model.add(GRU(90, 
    return_sequences=True,
    inner_activation='sigmoid',
    activation='tanh',
    truncate_gradient=clipval
    )
)
model.add(TimeDistributedDense(outputsize))
model.add(Activation('softmax'))

opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt)


res = pickle.load(gzip.open('models/samples_char_gru_res.pkl.gz','r'))
W = res['weights']
model.set_weights(W)


print('Sample sentence : ')

templist = [0.7, 1]
text_sampling_char(model,vocabs,templist,char='a',ns=200)