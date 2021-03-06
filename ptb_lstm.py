import cPickle as pickle
import gzip
from myutils import *
from train_utils import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils, generic_utils


outfile = 'results/ptb_lstm_out_small2.txt'
paramsfile = 'models/ptb_lstm_weights_small2.pkl.gz'
configfile = 'models/ptb_lstm_config_small2.pkl.gz'
print outfile,' ---- ', paramsfile

# configuration
learning_rate = 5e-2
seqlen = 35
batch_size = 50
embed_size = 200 #
hidden_size = 200 #small
max_epoch = 13
# clipval = 5
clipnorm = 5
embed_size = hidden_size


# load dataset
ptb = pickle.load(gzip.open('data/ptb_dataset.pkl.gz'))
word_to_id = ptb['word_to_id'] #dictionary of string -> int
train_data = ptb['train_data'] #list of integer
valid_data = ptb['valid_data'] #list of integer
test_data = ptb['test_data'] # list of integer
vocab_size = len(word_to_id) # integer


X, Y = vectorize_ptb(train_data, seqlen, vocab_size)
X_valid, Y_valid = vectorize_ptb(valid_data, seqlen, vocab_size)
X_test, Y_test = vectorize_ptb(test_data, seqlen, vocab_size)



# Yb = Y_train[0:20]
# Yh = to_onehot(Yb, vocab_size)

print('Build LSTM')
model = Sequential()

model.add(Embedding(vocab_size, embed_size))
model.add(LSTM(hidden_size,
	init='uniform',
	return_sequences=True,
	input_dim=embed_size)
)
model.add(LSTM(hidden_size,
	init='uniform',
	return_sequences=True)
)
model.add(TimeDistributedDense(vocab_size))
model.add(Activation('softmax'))

print 'Parameters: ', model.n_param

# opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6, clipvalue=clipval)
opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6, clipnorm=clipnorm)
model.compile(loss='categorical_crossentropy', optimizer=opt)


# Store configuration
res = {'config': model.get_config(),
    'seqlen':seqlen,
    'learning_rate':learning_rate,
    'batch_size':batch_size,
    'vocab_size':vocab_size,
    # 'clipval':clipval
    'clipnorm':clipnorm
}
pickle.dump(res, gzip.open(configfile,'w'))

train_rnn_ptb(model, X, Y, 
    X_valid, Y_valid, X_test, Y_test,
    batch_size=batch_size, iteration=max_epoch,vocab_size=vocab_size,
    outfile=outfile, paramsfile=paramsfile
) #see myutils.py








