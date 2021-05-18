# python -m wandb sweep --project "Deep-Learning-RNN" sweep.yaml
## python -m wandb agent "saish/Deep-Learning-RNN/dfv5qigk"

#python main.py  --epochs 5 --optimizer 'Adam' --Cell_Type 'GRU' --batch_size 64 --embedding_size 32 --n_enc_dec_layers 1 --hidden_layer_size 128 --dropout 0.2 --beam_size 1

# ------------------------------ #

# python -m wandb sweep --project "Deep-Learning-RNN" sweep_attention.yaml
# python -m wandb agent "saish/Deep-Learning-RNN/eftivn3u"
python attention_main.py  --epochs 1 --optimizer 'Adam' --l_rate 0.01 --Cell_Type 'GRU' --batch_size 64 --embedding_size 64 --n_enc_dec_layers 2 --hidden_layer_size 256 --dropout 0.0 --beam_size 10


