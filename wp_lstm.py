from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2


import numpy as np
import sys
import time

from myutils import *
import cPickle as pickle
import gzip


# Outputs
t = 2
outfile = 'results/wp_lstm_out'+str(t)+'.txt'
paramsfile = 'models/wp_lstm_weights'+str(t)+'.pkl.gz'
configfile = 'models/wp_lstm_config'+str(t)+'.pkl.gz'
print outfile,' ---- ', paramsfile


# hyper-parameters
seqlen = 100 # 
learning_rate = 2e-3
batch_size = 100
lettersize = 87
clipval = 50 # -1 : no clipping

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
print 'Corpus length: ', len(text), ', # vocabulary: ', inputsize, ', # '


print('Vectorization...')
X = np.zeros((n, seqlen), dtype='float32')
Y = np.zeros((n, seqlen, inputsize), dtype='float32')

for i, sent in enumerate(sents):
    prev_char = '*'
    for t in range(seqlen):
        char = sent[t]
        # print(prev_char ,' --- ', char)
        X[i, t] = char_indices[prev_char]
        Y[i, t, char_indices[char]] = 1
        prev_char = char

# ############

# build the model: 2 stacked LSTM
print('Build LSTM...')
model = Sequential()
model.add(Embedding(inputsize, lettersize))
model.add(LSTM(64, 
    return_sequences=True, 
    truncate_gradient=clipval, 
    input_dim=inputsize)
)
# model.add(Dropout(0.2))
model.add(LSTM(64, 
    return_sequences=True, 
    truncate_gradient=clipval)
)
model.add(TimeDistributedDense(outputsize))
model.add(Activation('softmax'))

opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
# opt = SGD(lr=learning_rate, momentum=0.9, decay=0.95, nesterov=True)
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