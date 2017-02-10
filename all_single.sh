!#/bin/bash
for iter in {1..10}
do
    for t in {1..20}
    do
        echo "Beginning iter $iter, task $t. training" | tee -a results/cnn/post_time_fix/run_single_$iter.txt
        python single.py --task_id $t --epochs 100 --random_state $((t*iter)) | tee -a results/cnn/post_time_fix/run_single_$iter.txt
    done 
done
