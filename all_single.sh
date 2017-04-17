!#/bin/bash
for iter in {1..1}
do
    for ((lm_e=0; lm_e<=10; lm_e += 1))
    do
        echo "Beginning iter $iter, lm epochs $lm_e. training"
        python single.py --task_id 1 --lm_epochs $lm_e
    done 
done
