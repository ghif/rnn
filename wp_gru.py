'''
    Text generation using GRU on War and Peace dataset
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

import numpy as np
import sys

from myutils import *

import cPickle as pickle
import gzip

# Outputs
t = 2
outfile = 'results/wp_gru_out'+str(t)+'.txt'
paramsfile = 'models/wp_gru_weights'+str(t)+'.pkl.gz'
configfile = 'models/wp_gru_config'+str(t)+'.pkl.gz'
print outfile,' ---- ', paramsfile

# hyper-parameters
seqlen = 100 # 
learning_rate = 8e-3
batch_size = 100
lettersize = 87
clipval = 50 # -1 : no clipping

# Data I/O
vocabs = initvocab_split('data/warpeace_input.txt', seqlen)
text = vocabs['text']
sents = vocabs['sents']
text_valid = vocabs['text_valid']
sents_valid = vocabs['sents_valid']
text_test = vocabs['text_test']
sents_test = vocabs['sents_test']

vocab = vocabs['vocab']
char_indices = vocabs['char_indices']
indices_char = vocabs['indices_char']

inputsize = len(vocab)
outputsize = inputsize


print('Vectorization...')

# Train
n = len(sents)
X = np.zeros((n, seqlen), dtype='float32')
Y = np.zeros((n, seqlen, inputsize), dtype='float32')

for i, sent in enumerate(sents):
    prev_char = '*'
    for t in range(seqlen):
        char = sent[t]
        X[i, t] = char_indices[prev_char]
        Y[i, t, char_indices[char]] = 1
        prev_char = char

# Valid
n_valid = len(sents_valid)
X_valid = np.zeros((n_valid, seqlen), dtype='float32')
Y_valid = np.zeros((n_valid, seqlen, inputsize), dtype='float32')
for i, sent in enumerate(sents_valid):
    prev_char = '*'
    for t in range(seqlen):
        char = sent[t]
        X_valid[i, t] = char_indices[prev_char]
        Y_valid[i, t, char_indices[char]] = 1
        prev_char = char


# Test
n_test = len(sents_test)
X_test = np.zeros((n_test, seqlen), dtype='float32')
Y_test = np.zeros((n_test, seqlen, inputsize), dtype='float32')
for i, sent in enumerate(sents_test):
    prev_char = '*'
    for t in range(seqlen):
        char = sent[t]
        X_test[i, t] = char_indices[prev_char]
        Y_test[i, t, char_indices[char]] = 1
        prev_char = char

# ############
print('Build GRU...')
model = Sequential()
model.add(Embedding(inputsize, lettersize))


model.add(GRU(100, 
    return_sequences=True, 
    inner_activation='sigmoid',
    activation='tanh'
    )
)
# model.add(Dropout(0.2))
# model.add(GRU(100, 
#     return_sequences=True,
#     inner_activation='sigmoid',
#     activation='tanh'
#     )
# )
# model.add(Dropout(0.2))
model.add(TimeDistributedDense(outputsize))
model.add(Activation('softmax'))

opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6, clipvalue=clipval)
model.compile(loss='categorical_crossentropy', optimizer=opt)


# Store configuration
res = {'config': model.get_config(),
    'seqlen':seqlen,
    'learning_rate':learning_rate,
    'batch_size':batch_size,
    'lettersize':lettersize,
    'clipval':clipval
}
pickle.dump(res, gzip.open(configfile,'w'))


train_rnn2(model, vocabs, X, Y, 
    X_valid, Y_valid, X_test, Y_test,
    batch_size=batch_size, iteration=50,
    outfile=outfile, paramsfile=paramsfile
) #see myutils.py