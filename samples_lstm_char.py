from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, generic_utils


import numpy as np
import sys

from myutils import *



# Data I/O
datapath = 'data/samples.txt'
text = open(datapath, 'r').read().lower()
chars = set(text) #vocab
data_size, vocab_size = len(text), len(chars)
print 'Corpus has %d characters, %d unique.' % (data_size, vocab_size)

outfile = 'sample_char_out.txt'
print(outfile)

# Char <-> Indices Mappings
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# hyper-parameters
maxlen = 7 # 
step = 1 # 
ns = 200 # number of samples


# Generate Data
sentences = []
# next_chars = []
next_sentences = []
for i in range(0, len(text) - maxlen - 1, step):
    sentences.append(text[i: i + maxlen])
    # next_chars.append(text[i + maxlen])
    next_sentences.append(text[i+1: i+maxlen+1])
print('nb sequences:', len(sentences))



print('Vectorization...')
n = len(sentences)
X = np.zeros((n, maxlen, vocab_size), dtype=np.bool)
Y = np.zeros((n, maxlen, vocab_size), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1


for i, sentence in enumerate(next_sentences):
    for t, char in enumerate(sentence):
        Y[i, t, char_indices[char]] = 1
    

# ############

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, vocab_size)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributedDense(vocab_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# outstr = ''
# fo = open(outfile,'w')
# fo.close()

def text_sampling(templist, ns=400):
    start_idx = np.random.randint(0, len(text) - maxlen - 1)

    outstr = ''
    for temperature in templist:
        print(' -- Temperature : ', temperature)
        outstr += ' -- Temperature : %f\n' % (temperature)
        
        char = text[start_idx]
        print('----- Generating with seed: "' + char + '"')
        outstr += ' -- Generating with seed : %s\n' % char

        generated = ''
        for iteration in range(ns):
            x = np.zeros((1, 1, len(chars)))
            x[0, 0, char_indices[char]] = 1
            # for t, char in enumerate(sentence):
            #     x[0, t, char_indices[char]] = 1.

            y = model.predict(x, verbose=0)
            print('x : ',x.shape)
            print(x)
            print('y : ',y.shape)
            print(y)
            next_index = sample(y, temperature)
            next_char = indices_char[next_index]

            outstr += next_char
            generated += next_char
            # sentence = sentence[1:] + next_char
            char = next_char

            
        print(generated)
        outstr += '\n\n'

    return outstr


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

    # fo = open(outfile,'a')
    # fo.write(outstr)    
    # fo.close()

    # print(' -- Training --')
    # model.fit(X,Y, batch_size=128, nb_epoch=1)