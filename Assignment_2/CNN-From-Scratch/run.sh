# export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
# export CUDA_VISIBLE_DEVICES=0

python main.py  --n_classes 10 --n_filters 32 --filter_multiplier 1 --filter_size 3 --epochs 5 --activation leakyrelu --batch_size 64 --batch_norm True --train_model True
