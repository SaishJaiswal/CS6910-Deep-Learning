#!/bin/sh

# For running sweeps
#python -m wandb sweep --project "project_name" sweep.yaml
#python -m wandb agent "saish/project_name/sweep_id"

#python -m wandb sweep --project "demo" sweep.yaml
#python -m wandb agent "saish/demo/gn0pd3id"

python main.py -l 784 128 32 10 --optimizer nadam --epochs 5 --l_rate 0.001 --batch_size 64 --loss cross_entropy --activation relu --n_hlayers 2 --hlayer_size 32
