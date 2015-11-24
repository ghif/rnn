'''
    Text generation using GRU on samples.txt

    - Looks good on iteration >= 15
    - The convergence rate is still much slower than that of Julia
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from extra_recurrent import LGRU


import numpy as np
import sys

from myutils import *

import cPickle as pickle
import gzip


# Outputs
outfile = 'results/samples_lgru_out.txt'
paramsfile = 'models/samples_lgru_weights.pkl.gz'
configfile = 'models/samples_lgru_config.pkl.gz'
print outfile,' ---- ', paramsfile

# hyper-parameters
seqlen = 50 # 
learning_rate = 5e-3
batch_size = 20
lettersize = 40
clipval = 5 # -1 : no clipping



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

model.add(LGRU(76, 
    return_sequences=True, 
    inner_activation='sigmoid',
    activation='tanh',
    truncate_gradient=clipval,
    input_dim=inputsize)
)
# model.add(Dropout(0.2))
model.add(LGRU(80, 
    return_sequences=True,
    inner_activation='sigmoid',
    activation='tanh',
    truncate_gradient=clipval
    )
)
# # model.add(Dropout(0.2))
model.add(LGRU(90, 
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
    batch_size=batch_size, iteration=500,
    outfile=outfile, paramsfile=paramsfile
) #see myutils.py

