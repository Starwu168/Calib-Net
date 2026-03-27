# How to train
## Conda settings
**requirements.txt+bpdist setup**
## Training
```
cd /path/to/your/calibnet && nohup setsid /data00/conda/miniconda3/envs/bpnet/bin/torchrun     --nproc_per_node=2     --master_port=11000     /path/to/your/calibnet/main.py     --config /path/to/your/calibnet/configs/calib_train.yaml     > /path/to/your/calibnet/train_nohup.log 2>&1 &
```
## Test
```
Calib-Net/val_vis.py --config /data00/wsx/code/calibnet/configs/calib_train.yaml --ckpt /data00/wsx/code/calibnet/runs_ZJU_loss_radarcam/calib_pmp_zju/best.pth --num 4097 --depth_max_list 50 70 80
```
