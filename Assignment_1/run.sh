#!/bin/sh

python main.py -l 784 128 64 10 --optimizer nadam --epochs 5 --l_rate 0.001 --batch_size 64
