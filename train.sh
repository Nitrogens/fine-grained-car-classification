
for((i=0;i<3;i++))
do
    echo $i
    /home/xmj/anaconda3/envs/3090/bin/python train.py --gpu $1 --i $i
done