env THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 nohup python -i wp_lstm.py > wp_lstm_nohup_2layer256.out & 
# env THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python -i wp_lstm.py