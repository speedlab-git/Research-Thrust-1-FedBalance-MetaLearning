CUDA_VISIBLE_DEVICES=5 python fcrimnet.py \
  --train_dir /mnt/beegfs/home/mdzarifhossa2025/mdhossa/NSF/ucf-crime-dataset/Train \
  --test_dir  /mnt/beegfs/home/mdzarifhossa2025/mdhossa/NSF/ucf-crime-dataset/Test \
  --num_clients 5 \
  --rounds 100 \
  --local_epochs 1 \
  --batch_size 64 \
  --use_wandb \
  --lr 3e-4
