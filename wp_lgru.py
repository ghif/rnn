'''
    Text generation using GRU on samples.txt

    - Looks good on iteration >= 15
    - The convergence rate is still much slower than that of Julia
'''

from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from extra_recurrent import LGRU, LGRU_FF
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop


import numpy as np
import sys

from myutils import *

import cPickle as pickle
import gzip


# Outputs
# t = 3 good, but slow (up to 500 iterations)
# t = 4
# outfile = 'results/wp_lgru_out'+str(t)+'.txt'
# paramsfile = 'models/wp_lgru_weights'+str(t)+'.pkl.gz'
# configfile = 'models/wp_lgru_config'+str(t)+'.pkl.gz'
outfile = 'results/wp_lgru_out_3layer128.txt'
paramsfile = 'models/wp_lgru_weights_3layer128.pkl.gz'
configfile = 'models/wp_lgru_config_3layer128.pkl.gz'
print outfile,' ---- ', paramsfile

# t = 3
# hyper-parameters
# seqlen = 128 # 
# learning_rate = 7e-3
# batch_size = 40
# lettersize = 40
# clipval = 30 # -1 : no clipping

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

print('Build LGRU...')
model = Sequential()
# 402888

model.add(LGRU(178, 
    return_sequences=True, 
    init='uniform',
    inner_activation='sigmoid',
    activation='tanh',
    input_dim=inputsize
    )
)
# model.add(Dropout(0.2))
model.add(LGRU(129, 
    return_sequences=True, 
    init='uniform',
    inner_activation='sigmoid',
    activation='tanh'
    )
)

model.add(LGRU(162, 
    return_sequences=True, 
    init='uniform',
    inner_activation='sigmoid',
    activation='tanh'
    )
)
# # # model.add(Dropout(0.2))
# model.add(LGRU_FF(180, 
#     return_sequences=True,
#     inner_activation='sigmoid',
#     activation='tanh'
#     )
# )
model.add(TimeDistributedDense(outputsize))
model.add(Activation('softmax'))

# print 'Parameters: ', model.n_param

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