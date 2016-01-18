import numpy as np
import random
import gzip
import cPickle as pickle
import time

from sklearn import preprocessing
from keras.utils import np_utils, generic_utils


def create_train_val(X, Y, p=0.7):
    [n, t, d] = X.shape
    n_train = int(p*n)
    n_val = n - n_train


    inds = np.random.permutation(n)
    
    inds_train = inds[:n_train]
    inds_val = [i for i in inds if i not in set(inds_train)]
    X_train = X[inds_train,:,:].astype('float32')
    Y_train = Y[inds_train,:,:].astype('float32')

    X_val = X[inds_val,:,:].astype('float32')
    Y_val = Y[inds_val,:,:].astype('float32')
    return (X_train, Y_train, X_val, Y_val)


def sample(z, temperature=1.0):
    # helper function to sample an index from a probability array
    z = np.log(z) / temperature
    z = np.exp(z) / np.sum(np.exp(z))
    return np.argmax(np.random.multinomial(1, z, 1))


def initvocab(datapath, seqlen):
    text = open(datapath, 'r').read()

    vocab = set(text)
    # vocab.remove('\n')
    vocab = sorted(vocab)
    vocab = ['*'] + vocab


    sents = []
    step = 1
    nsent = len(text) / seqlen
    for i in range(nsent):
        i1 = i*seqlen
        i2 = i*seqlen + seqlen
        sents.append(text[i1:i2])

    char_indices = dict((c, i) for i, c in enumerate(vocab))
    indices_char = dict((i, c) for i, c in enumerate(vocab))

    vocabs = {'text':text,
              'sents':sents,
              'vocab':vocab,
              'char_indices': char_indices,
              'indices_char':indices_char}
    
    return vocabs

def initvocab_split(datapath, seqlen, valid_frac=0.1, test_frac=0.1):
    text_o = open(datapath, 'r').read()
    print '[initvocal_split] len(text_o) : ',len(text_o)

    train_frac = 1. - (valid_frac + test_frac)
    n_train = int(len(text_o) * train_frac)
    n_valid = int(len(text_o) * valid_frac)
    n_test = int(len(text_o) * test_frac)

    text = text_o[0:n_train]
    text_valid = text_o[n_train:n_train + n_valid]
    text_test = text_o[n_train+n_valid: n_train+n_valid+n_test]

    print '[initvocal_split] len(text_valid) : ',len(text_valid)
    print '[initvocal_split] len(text_test) : ',len(text_test)

    vocab = set(text_o)
    # vocab.remove('\n')
    vocab = sorted(vocab)
    vocab = ['*'] + vocab

    char_indices = dict((c, i) for i, c in enumerate(vocab))
    indices_char = dict((i, c) for i, c in enumerate(vocab))

    # train
    sents = []
    step = 1
    nsent = len(text) / seqlen
    for i in range(nsent):
        i1 = i*seqlen
        i2 = i*seqlen + seqlen
        sents.append(text[i1:i2])

    # valid
    sents_valid = []
    step = 1
    nsent = len(text_valid) / seqlen
    for i in range(nsent):
        i1 = i*seqlen
        i2 = i*seqlen + seqlen
        sents_valid.append(text_valid[i1:i2])

    # test
    sents_test = []
    step = 1
    nsent = len(text_test) / seqlen
    for i in range(nsent):
        i1 = i*seqlen
        i2 = i*seqlen + seqlen
        sents_test.append(text_test[i1:i2])

    

    vocabs = {'text':text,
              'text_valid': text_valid,
              'text_test': text_test,
              'sents':sents,
              'sents_valid':sents_valid,
              'sents_test':sents_test,
              'vocab':vocab,
              'char_indices': char_indices,
              'indices_char':indices_char}
    
    return vocabs

def to_onehot(X, vocab_size):
    # X: [n, seqlen]
    # Y
    [n, seqlen] = X.shape
    Y = np.zeros((n, seqlen, vocab_size), dtype='float32')

    Lout = X.tolist()
    for i in xrange(n):
        Lin = Lout[i]
        

        for t, el in enumerate(Lin):
            Y[i, t, el] = 1

    return Y




def vectorize(vocabs, seqlen):
    sents = vocabs['sents']
    sents_valid = vocabs['sents_valid']
    sents_test = vocabs['sents_test']

    vocab = vocabs['vocab']
    char_indices = vocabs['char_indices']

    inputsize = len(vocab)

    def to_onehot(sents, seqlen, inputsize):

        n = len(sents)
        X = np.zeros((n, seqlen, inputsize), dtype='float32')
        Y = np.zeros((n, seqlen, inputsize), dtype='float32')

        for i, sent in enumerate(sents):
            prev_char = '*'
            for t in range(seqlen):
                char = sent[t]
                X[i, t, char_indices[prev_char]] = 1
                Y[i, t, char_indices[char]] = 1
                prev_char = char
        return X, Y

    X_train, Y_train = to_onehot(sents, seqlen, inputsize)
    X_valid, Y_valid = to_onehot(sents_valid, seqlen, inputsize)
    X_test, Y_test = to_onehot(sents_test, seqlen, inputsize)
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def vectorize_ptb(wordint_list, seqlen, inputsize):
    print('vectorize ptb')
    n = len(wordint_list)
    m = n / seqlen
    X = np.zeros((m, seqlen), dtype='int32')
    Y = np.zeros((m, seqlen), dtype='int32')
    # print(n-1)
    # print(seqlen)
    for i in xrange(m):
        # print(i)
        # word = wordint_list[i]
        for t in xrange(seqlen):
            X[i, t] = wordint_list[i*seqlen + t]
            Y[i, t] = wordint_list[(i*seqlen + t)+1]
        #     word = next_word
    return X, Y

def text_sampling_char(
    model,vocabs,templist,
    char='',ns=200):
    
    sents = vocabs['sents']
    vocab = vocabs['vocab']
    text = vocabs['text']
    char_indices = vocabs['char_indices']
    indices_char = vocabs['indices_char']

    inputsize = len(vocab)
    
    i = np.random.randint(0, len(sents))
    sent = sents[i]
    seqlen = len(sent)

    if not char:
        start_idx = np.random.randint(0, len(sent))
        char = sent[start_idx]


    outstr = ''

    for temperature in templist:
        
        print(' -- Temperature : ', temperature)
        outstr += ' -- Temperature : %f\n' % (temperature)
        
        
        
        print('----- Generating with seed: "' + char + '"')
        outstr += ' -- Generating with seed : %s\n' % char    

        generated = ''

        #### temperature ########
        sentences = np.zeros((1, ns, inputsize))
        sentences[0, 0, char_indices[char]] = 1
        # sentences = np.zeros((1, ns))
        # sentences[0, 0] = char_indices[char]
        
        for i in range(ns-1):
            y = model.predict(sentences, verbose=0)[0,i,:]            
            next_idx = sample(y, temperature)
            
            sentences[0, i+1, next_idx] = 1
            # sentences[0, i+1] = next_idx
            next_char = indices_char[next_idx]            
            generated += next_char

        ###########################


        print(generated)
        outstr += generated


        outstr += '\n\n'

    ### Print ARGMAX ####
    print ' -- ARGMAX -- '
    outstr += ' -- ARGMAX --\n'
    print('----- Generating with seed: "' + char + '"')
    outstr += ' -- Generating with seed : %s\n' % char    

    generated = ''

    sentences = np.zeros((1, ns, inputsize))
    sentences[0, 0, char_indices[char]] = 1
    # sentences = np.zeros((1, ns))
    # sentences[0, 0] = char_indices[char]
    
    for i in range(ns-1):
        y = model.predict(sentences, verbose=0)[0,i,:]            
        next_idx = np.argmax(y)
        
        sentences[0, i+1, next_idx] = 1
        # sentences[0, i+1] = next_idx
        next_char = indices_char[next_idx]            
        generated += next_char

    
    print(generated)
    outstr += generated
    outstr += '\n\n'


    return outstr

def perplexity(Y, P):
    [batch_size, seqlen, _] = Y.shape
    logp = np.log2(P)

    # == avoid nan ===
    pos = np.where(np.isinf(logp) == True)
    logp[pos] = -200
    # ======

    ppl = (-np.sum(np.multiply(Y, logp)) /  (seqlen * batch_size)) ** 2
    return ppl

def word_perplexity(loss):
    return np.exp(loss)


