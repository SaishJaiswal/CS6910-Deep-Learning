#!/bin/sh

python main.py -l 784 128 64 10 --optimizer adam --epochs 5 --l_rate 0.000001 --batch_size 64
