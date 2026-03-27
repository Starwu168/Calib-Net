#how to train
##conda settings
**requirements.txt+bpdist setup**
##training
**cd /path/to/your/calibnet && nohup setsid /data00/conda/miniconda3/envs/bpnet/bin/torchrun     --nproc_per_node=2     --master_port=11000     /path/to/your/calibnet/main.py     --config /path/to/your/calibnet/configs/calib_train.yaml     > /path/to/your/calibnet/train_nohup.log 2>&1 &**
