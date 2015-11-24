import theano
import theano.tensor as T
import numpy as np

x = T.vector("inputs")
w = T.dscalar("w")
v = T.dscalar("v")

def step(x_t, h_tm1, w, v):
	h_t = w * x_t + v * h_tm1
	return h_t


outputs_info = T.as_tensor_variable(np.asarray(0., x.dtype))
results, updates = theano.scan(fn=step,
			sequences=x,
			outputs_info=outputs_info,
			non_sequences=[w,v]
)

rnn = theano.function(inputs=[x], outputs=results)

# test
x = np.asarray([1,4,2,6])
w = 2
v = 3

print(x)

rnn(x, 2, 3)








