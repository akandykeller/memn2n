!#/bin/bash
for iter in {1..10}
do
    for t in {1..20}
    do
        echo "Beginning iter $iter, task $t. training" 
        python single.py --task_id $t --data_dir data/tasks_1-20_v1-2/en-10k/ 
    done 
done
