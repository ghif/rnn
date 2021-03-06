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
# t = 6 cross-entropy loss: ~1.5
# t = 7 # the current best for L1 = 64
# outfile = 'results/wp_lstm_out'+str(t)+'.txt'
# paramsfile = 'models/wp_lstm_weights'+str(t)+'.pkl.gz'
# configfile = 'models/wp_lstm_config'+str(t)+'.pkl.gz'

outfile = 'results/wp_lstm_out_3layer512_dropout0.2.txt'
paramsfile = 'models/wp_lstm_weights_3layer512_dropout0.2.pkl.gz'
configfile = 'models/wp_lstm_config_3layer512_dropout0.2.pkl.gz'
print outfile,' ---- ', paramsfile


# hyper-parameters
seqlen = 100 # 
learning_rate = 6e-3
batch_size = 50
clipval = 5 # -1 : no clipping

# Data I/O
vocabs = initvocab_split('data/warpeace_input.txt', seqlen)

vocab = vocabs['vocab']
inputsize = len(vocab)
outputsize = inputsize

print('Vectorization...')
X, Y, X_valid, Y_valid, X_test, Y_test = vectorize(vocabs, seqlen)

# ############

# build the model: 2 stacked LSTM
print('Build LSTM...')
model = Sequential()
# model.add(Embedding(inputsize, lettersize))
model.add(LSTM(512, 
    init='uniform',
    return_sequences=True,
    input_dim=inputsize)
)
model.add(Dropout(0.2))

model.add(LSTM(512, 
    init='uniform',
    return_sequences=True
    )
)
model.add(Dropout(0.2))

model.add(LSTM(512, 
    init='uniform',
    return_sequences=True
    )
)
model.add(Dropout(0.2))

model.add(TimeDistributedDense(outputsize))
model.add(Activation('softmax'))


print 'Parameters: ', model.n_param

opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6, clipvalue=clipval)
model.compile(loss='categorical_crossentropy', optimizer=opt)


# Store configuration
res = {'config': model.get_config(),
    'seqlen':seqlen,
    'learning_rate':learning_rate,
    'batch_size':batch_size,
    'lettersize':inputsize,
    'clipval':clipval
}
pickle.dump(res, gzip.open(configfile,'w'))

train_rnn2(model, vocabs, X, Y, 
    X_valid, Y_valid, X_test, Y_test,
    batch_size=batch_size, iteration=50,
    outfile=outfile, paramsfile=paramsfile
) #see myutils.py