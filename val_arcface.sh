s=(64)
alpha=(0.5)
for a in ${s[*]}; do
    for b in ${alpha[*]}; do
        for((i=0;i<3;i++)) do
            echo $a $b $i
            /home/xmj/anaconda3/envs/3090/bin/python val_arcface.py --gpu $1 --i ${i} --load_path "/home/xmj/ml-course/car/experiments/arcface-model-model_arcface-64-50-0.1-0.1-20-FocalLoss-${a}.0-0.5-False-${b}-2.0"
        done
        /home/xmj/anaconda3/envs/3090/bin/python calc_avg.py --val_file "/home/xmj/ml-course/car/experiments/arcface-model-model_arcface-64-50-0.1-0.1-20-FocalLoss-${a}.0-0.5-False-${b}-2.0/evaluation/all.log" --save_path "/home/xmj/ml-course/car/experiments/arcface-model-model_arcface-64-50-0.1-0.1-20-FocalLoss-${a}.0-0.5-False-${b}-2.0/evaluation/all_avg.log"
    done
done