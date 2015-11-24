import theano
import theano.tensor as T
import numpy as np

# test
_x = np.asarray([1,4,2,6], dtype='float32')
_w = 2
_v = 3
_u = 1

print(_x)

x = T.vector("inputs", dtype='float32')
w = T.scalar("w", dtype='float32')
v = T.scalar("v", dtype='float32')
u = T.scalar("u", dtype='float32')
h = T.scalar("outputs", dtype='float32')



def step(x_tm1, x_t,
	h_tm1, 
	w, v):
	
	h_t = w * x_t + v * h_tm1 + u * x_tm1
	return h_t


# zero left-padding
z = T.zeros_like(x)
y = T.concatenate(([z[-1]], x), axis=0)


results, updates = theano.scan(fn=step,
			sequences=[dict(input=y, taps=[-1, -0])],
			outputs_info=dict(initial=h, taps=[-1]),
			non_sequences=[w, v]			
)

rnn = theano.function(inputs=[x, h, w, v, u], outputs=results)



print rnn(_x, 0., _w, _v, _u)








