nohup python simple_train.py \
--save_dir ../data \
--train_ratio 0.8 \
--batch_size 64 \
--num_workers 4 \
--fc1 1024 \
--fc2 128 \
--fc_dropout 0.5 \
--optimizer Adam \
--lr 1e-3 \
--loss cross_entropy \
> "../logs/simple_train.log" 2>&1 &