#/bin/bash

for learning_rate in 0.1 1 10
do
    for dropout in 0.1 0.2
    do 
        for batch_size in 32 64
        do
            echo "-----------------------------------------------------------------------------------------" >> log.txt
            echo "learning_rate: $learning_rate, dropout: $dropout, batch_size: $batch_size" >> log.txt
            echo "-----------------------------------------------------------------------------------------" >> log.txt
            python3 main.py --lr $learning_rate --dropout $dropout --batch_size $batch_size
        done
        wait
    done
done