env THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 nohup python -i wp_gru.py > wp_gru_nohup_3layer256_dropout0.2.out &
# env THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python -i wp_gru.py