'''
    Text generation using GRU on War and Peace dataset
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
# from keras.initializations import uniform

import numpy as np
import sys

from myutils import *

import cPickle as pickle
import gzip

# Outputs
# t = 6 the best configuration so far
# t = 7 #same with 6, but with dropout
# t = 8 # remove the embedding layer
# t = 9 
# t = 10 # with roughly the same parameter as LSTM
# t = 11 # uniform initialization [-0.08, 0.08]

# outfile = 'results/wp_gru_out'+str(t)+'.txt'
# paramsfile = 'models/wp_gru_weights'+str(t)+'.pkl.gz'
# configfile = 'models/wp_gru_config'+str(t)+'.pkl.gz'

outfile = 'results/wp_gru_out_3layer128.txt'
paramsfile = 'models/wp_gru_weights_3layer128.pkl.gz'
configfile = 'models/wp_gru_config_3layer128.pkl.gz'
print outfile,' ---- ', paramsfile

# hyper-parameters
seqlen = 100 # 
learning_rate = 6e-3
batch_size = 50
# lettersize = 87
clipval = 5 # -1 : no clipping

# Data I/O
vocabs = initvocab_split('data/warpeace_input.txt', seqlen)

vocab = vocabs['vocab']
inputsize = len(vocab)
outputsize = inputsize

print('Vectorization...')
X, Y, X_valid, Y_valid, X_test, Y_test = vectorize(vocabs, seqlen)

# ############
print('Build GRU...')
model = Sequential()
# model.add(Embedding(inputsize, lettersize))


model.add(GRU(152, 
    return_sequences=True, 
    init='uniform',
    inner_activation='sigmoid',
    activation='tanh',
    input_dim=inputsize
    )
)
# model.add(Dropout(0.4))
model.add(GRU(147, 
    return_sequences=True, 
    init='uniform',
    inner_activation='sigmoid',
    activation='tanh'
    )
)

model.add(GRU(148, 
    return_sequences=True, 
    init='uniform',
    inner_activation='sigmoid',
    activation='tanh'
    )
)
# model.add(Dropout(0.4))
model.add(TimeDistributedDense(outputsize))
model.add(Activation('softmax'))

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