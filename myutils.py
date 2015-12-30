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


def train_rnn(model, vocabs,
    X, Y, batch_size=20, iteration=500, 
    outfile='outfile.txt', paramsfile='paramsfile.pkl.gz'):

    fo = open(outfile,'w')
    fo.close()

    losses = []
    ppl_avgs = []
    ppl_meds = []
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
        ppls = []

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
            ppl = perplexity(Y_batch, probs)
            ppls.append(ppl)


        elapsed_time = time.time() - start_time
        elapsed_times.append(elapsed_time)


        loss_avg = loss_avg / n_batches


        ppl_avg = np.average(ppls)
        print ''
        print '-- (Averaged) Perplexity : ',ppl_avg
        outstr += '-- (Averaged) Perplexity : %s\n' % ppl_avg
        ppl_avgs.append(ppl_avg)

        ppl_med = np.median(ppls)
        print '-- (Median) Perplexity : ',ppl_med
        outstr += '-- (Median) Perplexity : %s\n' % ppl_med
        ppl_meds.append(ppl_med)

        print '-- (Averaged) train loss : ',loss_avg
        outstr += '-- (Averaged) train loss : %s\n' % loss_avg
        losses.append(loss_avg)

        # store the training progress incl. the generated text
        fo = open(outfile,'a')
        fo.write(outstr)    
        fo.close()


        # store the other numerical results
        res = {'losses':losses, 
                'ppl_avgs':ppl_avgs,
                'ppl_meds':ppl_meds,
                'elapsed_times':elapsed_times,
                'weights': model.get_weights()
        }
        
        pickle.dump(res, gzip.open(paramsfile,'w'))

    return res

def train_rnn2(model, vocabs,
    X, Y, X_valid, Y_valid, X_test, Y_test,
    batch_size=100, iteration=50, 
    outfile='outfile.txt', paramsfile='paramsfile.pkl.gz'):

    fo = open(outfile,'w')
    fo.close()

    losses = []
    ppl_avgs = []
    ppl_meds = []

    losses_valid = []
    ppl_valid_avgs = []
    ppl_valid_meds = []

    losses_test = []
    ppl_test_avgs = []
    ppl_test_meds = []


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


        print(' == Training ==')
        
        start_time = time.time()

        loss_avg = 0.
        loss_list = []

        ppl = 0. #perplexity
        ppls = []

        n_batches = 0
        

        progbar = generic_utils.Progbar(X.shape[0])
        for X_batch, Y_batch in iterate_minibatches(X, Y, batch_size, shuffle=False):
            train_score = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("train loss", train_score)])


            # log loss
            # loss_avg += train_score
            loss_list.append(train_score)

            n_batches += 1

            # perplexity
            probs = model.predict(X_batch)
            ppl = perplexity(Y_batch, probs)
            ppls.append(ppl)


        elapsed_time = time.time() - start_time
        elapsed_times.append(elapsed_time)


        # loss_avg = loss_avg / n_batches
        print ''
        loss_avg = np.average(loss_list)
        print '-- (Averaged) train loss : ',loss_avg
        outstr += '-- (Averaged) train loss : %s\n' % loss_avg
        losses.append(loss_avg)

        ppl_avg = np.average(ppls)        
        print '-- (Averaged) Perplexity : ',ppl_avg
        outstr += '-- (Averaged) Perplexity : %s\n' % ppl_avg
        ppl_avgs.append(ppl_avg)

        ppl_med = np.median(ppls)
        print '-- (Median) Perplexity : ',ppl_med
        outstr += '-- (Median) Perplexity : %s\n' % ppl_med
        ppl_meds.append(ppl_med)

        


        print(' == VALIDATION ==')

        loss_valid_avg = model.evaluate(X_valid, Y_valid, batch_size=1024)
        print '-- (Averaged) Valid loss : ',loss_valid_avg
        outstr += '-- (Averaged) Valid loss : %s\n' % loss_valid_avg
        losses_valid.append(loss_valid_avg)
        
        probs_valid = model.predict(X_valid)
        ppls_valid = perplexity(Y_valid, probs_valid)
        ppl_avg = np.average(ppls_valid)
        
        print '-- (Averaged) Validation Perplexity : ',ppl_avg
        outstr += '-- (Averaged) Validation Perplexity : %s\n' % ppl_avg
        ppl_valid_avgs.append(ppl_avg)
        

        ppl_med = np.median(ppls_valid)
        print '-- (Median) Validation Perplexity : ',ppl_med
        outstr += '-- (Median) Validation Perplexity : %s\n' % ppl_med
        ppl_valid_meds.append(ppl_med)



        print(' == TEST ==')
        loss_test_avg = model.evaluate(X_test, Y_test, batch_size=1024)
        print '-- (Averaged) Test loss : ',loss_test_avg
        outstr += '-- (Averaged) Test loss : %s\n' % loss_test_avg
        losses_test.append(loss_test_avg)



        probs_test = model.predict(X_test)
        ppls_test = perplexity(Y_test, probs_test)

        ppl_avg = np.average(ppls_test)
        print '-- (Averaged) Test Perplexity : ',ppl_avg
        outstr += '-- (Averaged) Test Perplexity : %s\n' % ppl_avg
        ppl_test_avgs.append(ppl_avg)

        ppl_med = np.median(ppls_test)
        print '-- (Median) Test Perplexity : ',ppl_med
        outstr += '-- (Median) Test Perplexity : %s\n' % ppl_med
        ppl_test_meds.append(ppl_med)
        

        # store the training progress incl. the generated text
        fo = open(outfile,'a')
        fo.write(outstr)    
        fo.close()


        # store the other numerical results
        res = {'losses':losses,
                'ppl_avgs':ppl_avgs,
                'ppl_meds':ppl_meds,
                'losses_valid':losses_valid,
                'ppl_valid_avgs':ppl_valid_avgs,
                'ppl_valid_meds':ppl_valid_meds,
                'losses_test':losses_test,
                'ppl_test_avgs':ppl_test_avgs,
                'ppl_test_meds':ppl_test_meds,
                'elapsed_times':elapsed_times,
                'weights': model.get_weights()
        }
        
        pickle.dump(res, gzip.open(paramsfile,'w'))

    return res