#!/bin/sh

python main.py -l 784 128 32 10 --optimizer nag --epochs 5 --l_rate 0.001 --batch_size 64 --loss cross_entropy
