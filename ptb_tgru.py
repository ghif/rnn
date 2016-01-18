import cPickle as pickle
import gzip
from myutils import *
from train_utils import *
from keras.models import Sequential
from keras.initializations import uniform
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.embeddings import Embedding
from extra_recurrent import TGRU
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils, generic_utils


outfile = 'results/ptb_tgru_out_small_dropout0.2.txt'
paramsfile = 'models/ptb_tgru_weights_small_dropout0.2.pkl.gz'
configfile = 'models/ptb_tgru_config_small_dropout0.2.pkl.gz'
print outfile,' ---- ', paramsfile

# configuration
learning_rate = 1
lr_decay = 0.5
seqlen = 35
batch_size = 20
hidden_size = 250 #small
embed_size = hidden_size #
max_epoch = 13
decay_epoch = 4

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
model.add(TGRU(hidden_size,
	init='uniform',
    inner_init='uniform',
	return_sequences=True,
	input_dim=embed_size)
)
model.add(Dropout(0.2))
model.add(TGRU(hidden_size,
	init='uniform',
    inner_init='uniform',
	return_sequences=True)
)
model.add(Dropout(0.2))
model.add(TimeDistributedDense(vocab_size))
model.add(Activation('softmax'))

print 'Parameters: ', model.n_param

# opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6, clipvalue=clipval)
# opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6, clipnorm=clipnorm)
opt = SGD(lr=learning_rate, momentum=0.0, decay=lr_decay, clipnorm=clipnorm, decay_epoch=decay_epoch)
model.compile(loss='categorical_crossentropy', optimizer=opt)


# Store configuration
res = {'config': model.get_config(),
    'seqlen':seqlen,
    'learning_rate':learning_rate,
    'lr_decay':lr_decay,
    'batch_size':batch_size,
    'vocab_size':vocab_size,
    'max_epoch':max_epoch,
    'decay_epoch': decay_epoch,
    # 'clipval':clipval
    'clipnorm':clipnorm
}
pickle.dump(res, gzip.open(configfile,'w'))

train_rnn_ptb(model, X, Y, 
    X_valid, Y_valid, X_test, Y_test,
    batch_size=batch_size, iteration=max_epoch,vocab_size=vocab_size,
    outfile=outfile, paramsfile=paramsfile
) #see myutils.py








