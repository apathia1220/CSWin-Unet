# CSWin-Unet
# train
python train.py --dataset Synapse --cfg configs/cswin_tiny_224_lite.yaml --root_path "/mnt/HDD1/xxy/synapse/data/Synapse" --max_epochs 300 --output_dir "/mnt/HDD2/xxy/result/csunet_synapse_8"  --img_size 224 --base_lr 0.001 --batch_siz 24
# test
python test.py --dataset Synapse --cfg configs/cswin_tiny_224_lite.yaml --is_saveni --volume_path "/mnt/HDD1/xxy/synapse/data/Synapse" --output_dir /mnt/HDD2/xxy/result/csunet_synapse_8 --max_epoch 100 --base_lr 0.05 --img_size 224 --batch_size 24
