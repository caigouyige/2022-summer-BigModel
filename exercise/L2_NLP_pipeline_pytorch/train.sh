#/bin/bash

for learning_rate in 5 10 20
do
    for dropout in 0.1 0.2 0.5
    do 
        for batch_size in 32 64 96
        do
            echo "-----------------------------------------------------------------------------------------" >> log.txt
            echo "learning_rate: $learning_rate, dropout: $dropout, batch_size: $batch_size" >> log.txt
            echo "-----------------------------------------------------------------------------------------" >> log.txt
            python3 main.py --lr $learning_rate --dropout $dropout --batch_size $batch_size
        done
        wait
    done
done