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
t = 1
outfile = 'results/wp_gru_out'+str(t)+'.txt'
paramsfile = 'models/wp_gru_weights'+str(t)+'.pkl.gz'
configfile = 'models/wp_gru_config'+str(t)+'.pkl.gz'
print outfile,' ---- ', paramsfile

# hyper-parameters
seqlen = 100 # 
learning_rate = 2e-3
batch_size = 50
lettersize = 40
clipval = 5 # -1 : no clipping



# Data I/O
vocabs = initvocab('data/warpeace_input.txt', seqlen)
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

print('Build GRU...')
model = Sequential()
model.add(Embedding(inputsize, lettersize))

model.add(GRU(160, 
    return_sequences=True, 
    inner_activation='sigmoid',
    activation='tanh'
    )
)
# model.add(Dropout(0.2))
model.add(GRU(170, 
    return_sequences=True,
    inner_activation='sigmoid',
    activation='tanh'
    )
)
# # model.add(Dropout(0.2))
model.add(GRU(180, 
    return_sequences=True,
    inner_activation='sigmoid',
    activation='tanh'
    )
)
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


train_rnn(model, vocabs, X, Y, 
    batch_size=batch_size, iteration=50,
    outfile=outfile, paramsfile=paramsfile
) #see myutils.py