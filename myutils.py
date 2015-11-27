import numpy as np
import random
import gzip
import cPickle as pickle
import time

from sklearn import preprocessing
from keras.utils import np_utils, generic_utils



def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


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
        # sentences = np.zeros((1, ns, inputsize))
        # sentences[0, 0, char_indices[char]] = 1
        sentences = np.zeros((1, ns))
        sentences[0, 0] = char_indices[char]
        
        for i in range(ns-1):
            y = model.predict(sentences, verbose=0)[0,i,:]            
            next_idx = sample(y, temperature)
            
            sentences[0, i+1] = next_idx
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

    sentences = np.zeros((1, ns))
    sentences[0, 0] = char_indices[char]
    
    for i in range(ns-1):
        y = model.predict(sentences, verbose=0)[0,i,:]            
        next_idx = np.argmax(y)
        
        sentences[0, i+1] = next_idx
        next_char = indices_char[next_idx]            
        generated += next_char

    
    print(generated)
    outstr += generated
    outstr += '\n\n'




    return outstr

def perplexity(Y, P):
    [batch_size, seqlen, _] = Y.shape
    ppl = (-np.sum(np.multiply(Y, np.log2(P))) /  (seqlen * batch_size)) ** 2
    return ppl


def train_rnn(model, vocabs,
    X, Y, batch_size=20, iteration=500, 
    outfile='outfile.txt', paramsfile='paramsfile.pkl.gz'):

    fo = open(outfile,'w')
    fo.close()

    losses = []
    ppls = []
    elapsed_times = []
    

    for itr in range(1, iteration+1):
        print()
        outstr = ''
        print('*' * 50)
        outstr += '*******\n'
        
        print('Iteration', itr)
        outstr += 'Iteration : %d\n' % (itr)

        
        print('*' * 50)
        outstr += '*******\n'

        if itr % 10 == 0:
            print(' -- Text sampling ---')
            temperatures = [0.7, 1]
            generated = text_sampling_char(
                model,vocabs,
                temperatures, 
                ns=400)
            
            outstr += generated


        print(' -- Training --')
        
        start_time = time.time()

        loss_avg = 0.
        ppl = 0. #perplexity

        n_batches = 0

        progbar = generic_utils.Progbar(X.shape[0])
        for X_batch, Y_batch in iterate_minibatches(X, Y, batch_size, shuffle=False):
            train_score = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("train loss", train_score)])


            # log loss
            loss_avg += train_score
            n_batches += 1

            # perplexity
            probs = model.predict(X_batch)
            ppl += perplexity(Y_batch, probs)


        elapsed_time = time.time() - start_time
        elapsed_times.append(elapsed_time)


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
                'elapsed_times':elapsed_times,
                'weights': model.get_weights()
        }
        
        pickle.dump(res, gzip.open(paramsfile,'w'))

    return res