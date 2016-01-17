import cPickle as pickle
import gzip
from myutils import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils, generic_utils


outfile = 'results/ptb_lstm_out_small.txt'
paramsfile = 'models/ptb_lstm_weights_small.pkl.gz'
configfile = 'models/ptb_lstm_config_small.pkl.gz'
print outfile,' ---- ', paramsfile

# configuration
learning_rate = 1e-2
seqlen = 35
batch_size = 20
embed_size = 200 #
hidden_size = 200 #small
max_epoch = 13
embed_size = hidden_size

# load dataset
ptb = pickle.load(gzip.open('data/ptb_dataset.pkl.gz'))
word_to_id = ptb['word_to_id'] #dictionary of string -> int
train_data = ptb['train_data'] #list of integer
valid_data = ptb['valid_data'] #list of integer
test_data = ptb['test_data'] # list of integer
vocab_size = len(word_to_id) # integer


X_train, Y_train = vectorize_ptb(train_data, seqlen, vocab_size)

print('Build LSTM')
model = Sequential()

model.add(Embedding(vocab_size, embed_size))
model.add(LSTM(hidden_size,
	init='uniform',
	return_sequence=True,
	input_dim=embed_size)
)
model.add(LSTM(hidden_size,
	init='uniform',
	return_sequence=True)
)
model.add(TimeDistributedDense(vocab_size))










