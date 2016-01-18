import numpy as np
import random
import gzip
import cPickle as pickle
import time

from sklearn import preprocessing
from keras.utils import np_utils, generic_utils
from myutils import *

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

def evaluate(model, X, Y, batch_size=1024, vocab_size=35):
    # X : (n, seqlen)
    # Y : (n, seqlen)
    loss_list = []
    ppl_list = []
    word_ppl_list = []
    progbar = generic_utils.Progbar(X.shape[0])
    for X_batch, Y_batch in iterate_minibatches(X, Y, batch_size, shuffle=False):
        Y_batch_onehot = to_onehot(Y_batch, vocab_size)
        test_score = model.test_on_batch(X_batch, Y_batch_onehot)
        progbar.add(X_batch.shape[0], values=[("test loss", test_score)])

        #log-loss
        loss_list.append(test_score)

        # perplexity
        probs = model.predict(X_batch)
        ppl = perplexity(Y_batch_onehot, probs)
        ppl_list.append(ppl)


        #word-level perplexity
        word_ppl = word_perplexity(test_score)
        word_ppl_list.append(word_ppl)

    loss = np.average(loss_list)
    ppl = np.average(ppl_list)
    word_ppl = np.average(word_ppl_list)
    return (loss, ppl, word_ppl)

def train_rnn_ptb(model, X, Y, X_valid, Y_valid, X_test, Y_test,
    batch_size=100, iteration=50, vocab_size=10000,
    outfile='outfile.txt', paramsfile='paramsfile.pkl.gz'):

    fo = open(outfile,'w')
    fo.close()

    losses = []
    ppl_avgs = []
    word_ppl_avgs = []
    ppl_meds = []

    losses_valid = []
    ppl_valid_avgs = []
    word_ppl_valid_avgs = []
    ppl_valid_meds = []

    losses_test = []
    ppl_test_avgs = []
    word_ppl_test_avgs = []
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

        # if itr % 10 == 0:
        #     print(' -- Text sampling ---')
        #     temperatures = [0.7, 1]
        #     generated = text_sampling_char(
        #         model,vocabs,
        #         temperatures, 
        #         ns=400)
            
        #     outstr += generated


        print(' == Training ==')
        
        start_time = time.time()

        loss_list = []
        ppl_list = []
        word_ppl_list = []
        progbar = generic_utils.Progbar(X.shape[0])
        for X_batch, Y_batch in iterate_minibatches(X, Y, batch_size, shuffle=False):
            Y_batch_onehot = to_onehot(Y_batch, vocab_size)
            train_score = model.train_on_batch(X_batch, Y_batch_onehot)
            progbar.add(X_batch.shape[0], values=[("train loss", train_score)])

            #log-loss
            loss_list.append(train_score)

            # perplexity
            probs = model.predict(X_batch)
            ppl = perplexity(Y_batch_onehot, probs)
            ppl_list.append(ppl)


            #word-level perplexity
            word_ppl = word_perplexity(train_score)
            word_ppl_list.append(word_ppl)


        elapsed_time = time.time() - start_time
        elapsed_times.append(elapsed_time)

        loss = np.average(loss_list)
        ppl = np.average(ppl_list)
        word_ppl = np.average(word_ppl_list)
        # (loss, ppl, word_ppl) = evaluate(model, X, Y, batch_size=1024, vocab_size=10000)
        print '\n'
        print '-- (Averaged) train loss : ',loss
        outstr += '-- (Averaged) train loss : %s\n' % loss
        losses.append(loss)

        print '-- (Averaged) Word-level Perplexity : ',word_ppl
        outstr += '-- (Averaged) Word-level Perplexity : %s\n' % word_ppl
        word_ppl_avgs.append(word_ppl)

        print '-- (Averaged) Perplexity : ',ppl
        outstr += '-- (Averaged) Perplexity : %s\n' % ppl
        ppl_avgs.append(ppl)


        print(' == VALIDATION ==')
        (loss, ppl, word_ppl) = evaluate(model, X_valid, Y_valid, batch_size=512, vocab_size=10000)
        print '\n'
        print '-- (Averaged) Valid loss : ',loss
        outstr += '-- (Averaged) Valid loss : %s\n' % loss
        losses_valid.append(loss)

        print '-- (Averaged) Valid Word-level Perplexity : ',word_ppl
        outstr += '-- (Averaged) Valid Word-level Perplexity : %s\n' % word_ppl
        word_ppl_valid_avgs.append(word_ppl)

        print '-- (Averaged) Validation Perplexity : ',ppl
        outstr += '-- (Averaged) Validation Perplexity : %s\n' % ppl
        ppl_valid_avgs.append(ppl)



        print(' == TEST ==')
        (loss, ppl, word_ppl) = evaluate(model, X_test, Y_test, batch_size=512, vocab_size=10000)
        print '\n'
        print '-- (Averaged) Test loss : ',loss
        outstr += '-- (Averaged) Test loss : %s\n' % loss
        losses_test.append(loss)

        print '-- (Averaged) Test Word-level Perplexity : ',word_ppl
        outstr += '-- (Averaged) Test Word-level Perplexity : %s\n' % word_ppl
        word_ppl_test_avgs.append(word_ppl)

        print '-- (Averaged) Test Perplexity : ',ppl
        outstr += '-- (Averaged) Test Perplexity : %s\n' % ppl
        ppl_test_avgs.append(ppl)


        # store the training progress incl. the generated text
        fo = open(outfile,'a')
        fo.write(outstr)    
        fo.close()


        # store the other numerical results
        res = {'losses':losses,
                'word_ppl_avgs':word_ppl_avgs,
                'ppl_avgs':ppl_avgs,
                'ppl_meds':ppl_meds,
                'losses_valid':losses_valid,
                'word_ppl_valid_avgs':word_ppl_valid_avgs,
                'ppl_valid_avgs':ppl_valid_avgs,
                'ppl_valid_meds':ppl_valid_meds,
                'losses_test':losses_test,
                'word_ppl_test_avgs':word_ppl_test_avgs,
                'ppl_test_avgs':ppl_test_avgs,
                'ppl_test_meds':ppl_test_meds,
                'elapsed_times':elapsed_times,
                'weights': model.get_weights()
        }
        
        pickle.dump(res, gzip.open(paramsfile,'w'))

    return res

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

            # n_batches += 1

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


