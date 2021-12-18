s=(16 32 64)
alpha=(0.5 1.0)
for a in ${s[*]}; do
    for b in ${alpha[*]}; do
        for((i=0;i<3;i++)) do
            echo $a $b $i
            /home/xmj/anaconda3/envs/3090/bin/python train_arcface.py --gpu $1 --i ${i} --s ${a} --alpha ${b}
        done
    done
done