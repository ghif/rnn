from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
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
chars = set(text)
data_size, vocab_size = len(text), len(chars)
print 'Corpus has %d characters, %d unique.' % (data_size, vocab_size)

outfile = 'sample_out.txt'

# Char <-> Indices Mappings
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# hyper-parameters
maxlen = 20 # sentence length, maxlen = 1 --> char-to-char, maxlen > 1 --> substring-to-char
step = 3 # 
ns = 400 # number of samples


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
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, vocab_size)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')



# iteration = 1

outstr = ''

fo = open(outfile,'w')
fo.close()

# while(True):
for iteration in range(1, 200):
    print()
    
    print('*' * 50)
    outstr += '*******\n'
    
    print('Iteration', iteration)
    outstr += 'Iteration : %d\n' % (iteration)

    
    print('*' * 50)
    outstr += '*******\n'

    model.fit(X,Y, batch_size=128, nb_epoch=1)

    print(' -- Text sampling ---')
    start_idx = np.random.randint(0, len(text) - maxlen -1)
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print(' -- Temperature : ', temperature)
        outstr += ' -- Temperature : %f\n' % (temperature)


        # sen_seed = text[start_idx: start_idx + maxlen]
        # print(' >> seed : ',sen_seed)
        # outstr += ' >> seed : %s \n' % (sen_seed)

        # txt = ''
        # for it in range(ns):
        #     x = np.zeros((1, maxlen, len(chars)))
        #     for t, char in enumerate(sen_seed):
        #         x[0, t, char_indices[char]] = 1

        #     y = model.predict(x, verbose=0)[0]
        #     next_index = sample(y, temperature)
        #     next_char = indices_char[next_index]
        #     txt += next_char


        # print '----\n %s \n----' % (txt, )
        # outstr += '----\n %s \n----\n' % (txt, )
        generated = ''
        sentence = text[start_idx: start_idx + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for iteration in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

            fo = open(outfile,'a')
            fo.write(outstr + generated)    
            fo.close()
        print()





    


    # iteration += 1
	





