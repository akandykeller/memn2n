!#/bin/bash
for iter in {1..10}
do
    for t in {1..20}
    do
        echo "Beginning iter $iter, task $t. training" | tee -a results_personal/rnn_adj_ae/run_1k_$iter.txt
        python single.py --task_id $t --epochs 20 --data_dir data/tasks_1-20_v1-2/en/ | tee -a results_personal/rnn_adj_ae/run_1k_$iter.txt
    done 
done
