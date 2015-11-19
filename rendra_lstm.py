from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, generic_utils


import numpy as np
import sys
import gzip
import cPickle as pickle

from myutils import *



# Data I/O
datapath = 'data/rendra.txt'
text = open(datapath, 'r').read().lower()
chars = set(text)
data_size, vocab_size = len(text), len(chars)
print 'Corpus has %d characters, %d unique.' % (data_size, vocab_size)

outfile = 'rendra_cho_out.txt'
print(outfile)

# Char <-> Indices Mappings
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# hyper-parameters
maxlen = 20 # sentence length, maxlen = 1 --> char-to-char, maxlen > 1 --> substring-to-char
step = 1 # 
ns = 400 # number of samples
batch_size = 128


sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
n = len(sentences)
X = np.zeros((n, maxlen, vocab_size), dtype=np.bool)
Y = np.zeros((n, vocab_size), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1

    Y[i, char_indices[next_chars[i]]] = 1



# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(maxlen, vocab_size)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

outstr = ''

fo = open(outfile,'w')
fo.close()

def text_sampling(templist, ns=400):
    start_idx = np.random.randint(0, len(text) - maxlen - 1)

    outstr = ''
    for temperature in templist:
        print(' -- Temperature : ', temperature)
        outstr += ' -- Temperature : %f\n' % (temperature)
        
        sentence = text[start_idx: start_idx + maxlen]
        print('----- Generating with seed: "' + sentence + '"')
        outstr += ' -- Generating with seed : %s\n' % sentence
        outstr += '=== Below is the generated text === \n'
        generated = '=== Below is the generated text === \n'
        for iteration in range(ns):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            y = model.predict(x, verbose=0)[0]
            next_index = sample(y, temperature)
            next_char = indices_char[next_index]

            outstr += next_char
            generated += next_char
            sentence = sentence[1:] + next_char

            
        print(generated)
        outstr += '\n\n'

    return outstr



losses = []
perplexities = []
for iteration in range(1, 1000):
    print()
    outstr = ''
    print('*' * 50)
    outstr += '*******\n'
    
    print('Iteration', iteration)
    outstr += 'Iteration : %d\n' % (iteration)

    
    print('*' * 50)
    outstr += '*******\n'


    print(' -- Text sampling ---')
    generated = text_sampling([0.2, 0.5, 1., 1.2], ns=ns)
    outstr += generated

    

    print(' -- Training --')
    # model.fit(X,Y, batch_size=128, nb_epoch=1)
    progbar = generic_utils.Progbar(X.shape[0])

    loss_avg = 0.
    ppl = 0. #perplexity


    n_batch = 0
    for X_batch, Y_batch in iterate_minibatches(X, Y, batch_size, shuffle=False):
        train_score = model.train_on_batch(X_batch, Y_batch)
        progbar.add(X_batch.shape[0], values=[("train loss", train_score)])
        loss_avg += train_score
        n_batch += 1

        # compute perplexity here
        


    loss_avg = loss_avg / n_batch
    print(' -- (Averaged) train loss : ',loss_avg)

    outstr += '-- (Averaged) train loss'
    losses.append(loss_avg)

    # store the training progress incl. the generated text
    fo = open(outfile,'a')
    fo.write(outstr)    
    fo.close()


    # store the other numerical results
    res = {'losses':losses}
    res = {'weights':model.get_weights()}
    pickle.dump(res, gzip.open('rendra_lstm_cho_res.pkl.gz','w'))
