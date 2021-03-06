from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from extra_recurrent import LGRU, LGRU2
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop


import numpy as np
import sys

from myutils import *

import cPickle as pickle
import gzip

print('=== LOAD MODEL ===')
t = 9 # 
outfile = 'results/samples_lgru_out'+str(t)+'.txt'
paramsfile = 'models/samples_lgru_weights'+str(t)+'.pkl.gz'
configfile = 'models/samples_lgru_config'+str(t)+'.pkl.gz'
print outfile,' ---- ', paramsfile

# hyper-parameters
seqlen = 50 # 
learning_rate = 5e-3
batch_size = 20
lettersize = 40
clipval = 5

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


print('Vectorization...')
X = np.zeros((n, seqlen), dtype='float32')
Y = np.zeros((n, seqlen, inputsize), dtype='float32')

for i, sent in enumerate(sents):
    prev_char = '*'
    for t in range(seqlen):
        char = sent[t]
        # print(prev_char ,' --- ', char)
        # X[i, t, char_indices[prev_char]] = 1
        X[i, t] = char_indices[prev_char]
        Y[i, t, char_indices[char]] = 1
        prev_char = char

# ############

print('Build LGRU...')
model = Sequential()
model.add(Embedding(inputsize, lettersize))

model.add(LGRU2(76, 
    return_sequences=True, 
    inner_activation='sigmoid',
    activation='tanh'
    )
)
# model.add(Dropout(0.2))
model.add(LGRU2(80, 
    return_sequences=True,
    inner_activation='sigmoid',
    activation='tanh'
    )
)
# # model.add(Dropout(0.2))
model.add(LGRU2(90, 
    return_sequences=True,
    inner_activation='sigmoid',
    activation='tanh'
    )
)
model.add(TimeDistributedDense(outputsize))
model.add(Activation('softmax'))

opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6, clipvalue=clipval)
model.compile(loss='categorical_crossentropy', optimizer=opt)


res = pickle.load(gzip.open(paramsfile,'r'))
W = res['weights']
model.set_weights(W)

print(' -- Text sampling ---')
temperatures = [0.7, 1]
generated = text_sampling_char(
    model,vocabs,
    temperatures, 
    ns=400)