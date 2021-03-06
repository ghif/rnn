from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from extra_recurrent import TGRU_CHECK
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop


import numpy as np
import sys

from myutils import *

import cPickle as pickle
import gzip


# Outputs
# t = 3 good, but slow (up to 500 iterations)
# t = 1
# outfile = 'results/wp_tlstm_out'+str(t)+'.txt'
# paramsfile = 'models/wp_tlstm_weights'+str(t)+'.pkl.gz'
# configfile = 'models/wp_tlstm_config'+str(t)+'.pkl.gz'
outfile = 'results/wp_tgru_check_out_3layer256_dropout2-2.txt'
paramsfile = 'models/wp_tgru_check_weights_3layer256_dropout2-2.pkl.gz'
configfile = 'models/wp_tgru_check_config_3layer256_dropout2-2.pkl.gz'
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

print('Build T-GRU...')
model = Sequential()
# 402888

model.add(TGRU_CHECK(459, 
    return_sequences=True, 
    init='uniform',
    inner_activation='sigmoid',
    activation='tanh',
    dropout=0.0,
    input_dim=inputsize
    )
)
# model.add(Dropout(0.5))

model.add(TGRU_CHECK(458, 
    return_sequences=True, 
    init='uniform',
    inner_activation='sigmoid',
    dropout=0.1,
    activation='tanh'
    )
)
# model.add(Dropout(0.5))

model.add(TGRU_CHECK(457, 
    return_sequences=True, 
    init='uniform',
    inner_activation='sigmoid',
    dropout=0.1,
    activation='tanh'
    )
)
# model.add(Dropout(0.5))

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