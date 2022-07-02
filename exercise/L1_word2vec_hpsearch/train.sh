#!/bin/bash

for model_name in cbow skipgram
do
    for train_batch_size in 50 100 150
    do
        for learing_rate in 0.25 0.50 1.00
        do 
            echo -n "Model name: $model_name, " >> log.txt
            echo -n "Train batch size = $train_batch_size, " >> log.txt
            echo -n "Learning rate = $learing_rate, " >> log.txt
            echo -n "Val Loss = " >> log.txt
            python3 edit.py $train_batch_size $learing_rate $model_name
            python3 train.py --config config.yaml
            echo -e -n '\n' >> log.txt
        done
        wait
    done
done