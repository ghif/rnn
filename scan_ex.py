import theano
import theano.tensor as T
import numpy as np

x = T.vector("inputs", dtype='float32')
w = T.scalar("w", dtype='float32')
v = T.scalar("v", dtype='float32')
# h = T.vector("outputs", dtype='float32')
h = T.scalar("outputs", dtype='float32')

def step(x_t, 
	h_tm1, 
	w, v):
	h_t = w * x_t + v * h_tm1
	return h_t


# outputs_info = T.as_tensor_variable(np.asarray(0., x.dtype))
# outputs_info = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
results, updates = theano.scan(fn=step,
			sequences=x,
			outputs_info=h,
			non_sequences=[w, v]			
)

rnn = theano.function(inputs=[x, h, w, v], outputs=results[-1])

# test
x = np.asarray([1,4,2,6], dtype='float32')
w = 2
v = 3

print(x)

print rnn(x, 0., w, v)








