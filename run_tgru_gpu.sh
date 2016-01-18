# env THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 nohup python -i wp_tgru.py > wp_tgru_nohup_3layer512_dropout0.2.out &
env THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 nohup python -i ptb_tgru.py > ptb_tgru_nohup_small_dropout0.2.out &
