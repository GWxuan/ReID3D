
TRAIN_TXT=../database/lreid/outputs/train_path.txt
TRAIN_INFO=../database/lreid/outputs/train_info.npy
TEST_TXT=../database/lreid/outputs/test_path.txt
TEST_INFO=../database/lreid/outputs/test_info.npy
QUERY_INFO=../database/lreid/outputs/query_IDX.npy
   
CKPT=./log
PORT=29537
python3 main.py \
    --train_txt $TRAIN_TXT --train_info $TRAIN_INFO  --test_batch 16 \
    --test_txt $TEST_TXT  --test_info $TEST_INFO --query_info $QUERY_INFO \
    --n_epochs 700 --lr 0.00005 --lr_step_size 30 --optimizer AdamW \
    --ckpt $CKPT --log_path loss.txt --num_workers 4\
    --class_per_batch 4 --track_per_class 4 --seq_len 30\
    --feat_dim 1024 --stride 1 --gpu_id '6,7' --eval_freq 1;
