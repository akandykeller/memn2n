!#/bin/bash
for iter in {1..10}
do
    for t in {1..20}
    do
        echo "Beginning iter $iter, task $t. training" | tee -a results/rnn_adj_all/run_$iter.txt
        python single.py --task_id $t --epochs 20 | tee -a results/rnn_adj_all/run_$iter.txt
    done 
done
