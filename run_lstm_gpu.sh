# env THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 nohup python -i wp_lstm.py > wp_lstm_nohup_1layer256.out & 
env THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python -i ptb_lstm.py
# env THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 nohup python -i ptb_lstm.py > ptb_lstm_nohup_small.out & 

