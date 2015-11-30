import theano
import theano.tensor as T
import numpy as np


from keras import activations, initializations
from keras.layers.recurrent import Recurrent
from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix

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
		self.input = T.tensor3()

		# forget gate params
		self.W_xf = self.init((input_dim, self.output_dim))
		self.U_hf = self.inner_init((self.output_dim, self.output_dim))
		self.b_f = shared_zeros((self.output_dim))

		# input/feature params
		self.W_xz = self.init((input_dim, self.output_dim))
		# self.U_xz = self.inner_init((self.output_dim, self.output_dim))
		self.U_xz = self.inner_init((input_dim, self.output_dim))
		self.b_z = shared_zeros((self.output_dim))

		# output params
		self.W_xo = self.init((input_dim, self.output_dim))
		# self.U_xo = self.inner_init((self.output_dim, self.output_dim))
		self.U_xo = self.inner_init((input_dim, self.output_dim))
		self.b_o = shared_zeros((self.output_dim))

		self.params = [
			self.W_xf, self.U_hf, self.b_f,
			self.W_xz, self.U_xz, self.b_z,
			self.W_xo, self.U_xo, self.b_o,
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	
	def _step(self,
              xf_t, xz_t, xo_t, mask_tm1, x_tm1,
              h_tm1, c_tm1,
              U_f, U_z, U_o):
		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1


		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, U_f))
		z_t = self.inner_activation(xz_t + T.dot(x_tm1, U_z))
		o_t = self.activation(xo_t + T.dot(x_tm1, U_o))
		c_t = f_t * c_mask_tm1 + (1 - f_t) * z_t
		h_t = c_t * o_t


		return [h_t, c_t]

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		# Create X_tm1 sequence through zero left-padding
		Z = T.zeros_like(X)
		X_tm1 = T.concatenate(([Z[0]], X), axis=0)

		
		x_f = T.dot(X, self.W_xf) + self.b_f
		x_z = T.dot(X, self.W_xz) + self.b_z
		x_o = T.dot(X, self.W_xo) + self.b_o

		h_info = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
		c_info = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

		[outputs, cells], updates = theano.scan(
		    self._step,
		    sequences=[x_f, x_z, x_o, padded_mask, X_tm1],
		    outputs_info=[h_info, c_info],
		    non_sequences=[self.U_hf, self.U_xz, self.U_xo],
		    truncate_gradient=self.truncate_gradient,
		    go_backwards=self.go_backwards)

		if self.return_sequences:
		    return outputs.dimshuffle((1, 0, 2))
		return outputs[-1]

	def get_config(self):
		config = {"name": self.__class__.__name__,
		          "output_dim": self.output_dim,
		          "init": self.init.__name__,
		          "inner_init": self.inner_init.__name__,
		          "activation": self.activation.__name__,
		          "inner_activation": self.inner_activation.__name__,
		          "truncate_gradient": self.truncate_gradient,
		          "return_sequences": self.return_sequences,
		          "input_dim": self.input_dim,
		          "input_length": self.input_length,
		          "go_backwards": self.go_backwards}
		base_config = super(LGRU, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


# Tester
if __name__ == "__main__":
	model = LGRU(100)