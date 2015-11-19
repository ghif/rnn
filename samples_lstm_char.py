from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, generic_utils


import numpy as np
import sys

from myutils import *
import cPickle as pickle
import gzip


# Outputs
outfile = 'sample4_char_out.txt'
print(outfile)
outparams = 'samples4_char_lstm_res.pkl.gz'

# hyper-parameters
seqlen = 50 # 
batch_size = 20

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
X = np.zeros((n, seqlen, inputsize), dtype='float32')
Y = np.zeros((n, seqlen, inputsize), dtype='float32')

for i, sent in enumerate(sents):
    prev_char = '*'
    for t in range(seqlen):
        char = sent[t]
        # print(prev_char ,' --- ', char)
        X[i, t, char_indices[prev_char]] = 1
        Y[i, t, char_indices[char]] = 1
        prev_char = char

# ############

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(76, 
    return_sequences=True, 
    truncate_gradient=5, 
    input_dim=inputsize)
)
# model.add(Dropout(0.2))
model.add(LSTM(80, 
    return_sequences=True, 
    truncate_gradient=5)
)
# model.add(Dropout(0.2))
model.add(LSTM(90, 
    return_sequences=True, 
    truncate_gradient=5)
)
model.add(TimeDistributedDense(outputsize))
model.add(Activation('softmax'))

opt = RMSprop(lr=1e-2, rho=0.9, epsilon=1e-6)
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

    if iteration % 10 == 0:
        print(' -- Text sampling ---')
        temperatures = [0.2, 0.5, 1., 1.2]
        generated = text_sampling_char(
            model,vocabs,
            temperatures, 
            ns=200)
        outstr += generated

    fo = open(outfile,'a')
    fo.write(outstr)    
    fo.close()

    print(' -- Training --')
    
    progbar = generic_utils.Progbar(X.shape[0])

    loss_avg = 0.
    ppl = 0. #perplexity

    n_batches = 0
    for X_batch, Y_batch in iterate_minibatches(X, Y, batch_size, shuffle=False):
        train_score = model.train_on_batch(X_batch, Y_batch)
        progbar.add(X_batch.shape[0], values=[("train loss", train_score)])
        loss_avg += train_score
        n_batches += 1

        # compute perplexity here
        probs = model.predict(X_batch)
        ppl += -np.sum(np.multiply(Y_batch, np.log2(probs))) / batch_size


    loss_avg = loss_avg / n_batches
    ppl = ppl / n_batches

    print '-- (Averaged) Perplexity : ',ppl
    outstr += '-- (Averaged) Perplexity : %s\n' % ppl
    ppls.append(ppl)

    print '-- (Averaged) train loss : ',loss_avg
    outstr += '-- (Averaged) train loss : %s\n' % loss_avg
    losses.append(loss_avg)

    # store the training progress incl. the generated text
    fo = open(outfile,'a')
    fo.write(outstr)    
    fo.close()


    # store the other numerical results
    res = {'losses':losses}
    res = {'ppls':ppls}
    res = {'weights':model.get_weights()}
    pickle.dump(res, gzip.open(outparams,'w'))