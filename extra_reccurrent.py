import theano
import theano.tensor as T
import numpy as np


from keras import activations, initializations
from keras.layers.recurrent import Recurrent

class LGRU(Recurrent):
	def __init__(self, output_dim,
		init='glorot_uniform', inner_init='orthogonal',
		activation='sigmoid', inner_activation='hard_sigmoid',
		weights=None, truncate_gradient=-1, return_sequences=False,
		input_dim=None, input_length=None, go_backwards=False, **kwargs):

		self.output_dim = output_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.initial_weights = weights
		self.go_backwards = go_backwards

		self.input_dim = input_dim
		self.input_length = input_length
		if self.input_dim:
		    kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(LGRU, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]

		


if __name__ == "__main__":
	model = LGRU(100)