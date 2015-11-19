'''
    Text generation using GRU on samples.txt

    - Looks good on iteration >= 40
    - The convergence rate is still much slower than that of Julia
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import GRU
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



# Outputs
outfile = 'sample2_char_gru_out.txt'
print(outfile)
outparams = 'samples2_char_gru_res.pkl.gz'

# hyper-parameters
seqlen = 50 # 
learning_rate = 6e-3
batch_size = 10
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

outstr = ''
fo = open(outfile,'w')
fo.close()



losses = []
ppls = []
for iteration in range(1, 500):
    print()
    outstr = ''
    print('*' * 50)
    outstr += '*******\n'
    
    print('Iteration', iteration)
    outstr += 'Iteration : %d\n' % (iteration)

    
    print('*' * 50)
    outstr += '*******\n'

    if iteration % 5 == 0:
        print(' -- Text sampling ---')
        temperatures = [0.7, 1]
        generated = text_sampling_char(
            model,vocabs,
            temperatures, 
            ns=200)
        
        outstr += generated

    fo = open(outfile,'a')
    fo.write(outstr)    
    fo.close()

    print(' -- Training --')
    

    loss_avg = 0.
    ppl = 0. #perplexity

    n_batches = 0

    progbar = generic_utils.Progbar(X.shape[0])
    for X_batch, Y_batch in iterate_minibatches(X, Y, batch_size, shuffle=False):
        # for t in range(seqlen):
        #     ix = np.argmax(X_batch[0,t,:])
        #     iy = np.argmax(Y_batch[0,t,:])

        #     print(indices_char[ix], '-- ',indices_char[iy])
            
        train_score = model.train_on_batch(X_batch, Y_batch)
        progbar.add(X_batch.shape[0], values=[("train loss", train_score)])


        # log loss
        loss_avg += train_score
        n_batches += 1

        # perplexity
        probs = model.predict(X_batch)
        ppl += perplexity(Y_batch, probs)




    loss_avg = loss_avg / n_batches
    ppl = ppl / n_batches

    print ''
    print '-- (Averaged) Perplexity : ',ppl
    outstr += '-- (Averaged) Perplexity : %s\n' % ppl
    ppls.append(ppl)
    outstr += '-- (Median) Perplexity : %s\n' % np.median(ppls)

    print '-- (Averaged) train loss : ',loss_avg
    outstr += '-- (Averaged) train loss : %s\n' % loss_avg
    losses.append(loss_avg)

    # store the training progress incl. the generated text
    fo = open(outfile,'a')
    fo.write(outstr)    
    fo.close()


    # store the other numerical results
    res = {'losses':losses, 
            'ppls':ppls,
            'weights': model.get_weights(),
            'config': model.get_config(),
            'seqlen':seqlen,
            'learning_rate':learning_rate,
            'batch_size':batch_size,
            'lettersize':lettersize,
            'clipval':clipval
    }
    
    pickle.dump(res, gzip.open(outparams,'w'))