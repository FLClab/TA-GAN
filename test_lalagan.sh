#!/bin/bash 

python test.py --dataroot DendriticFActinDataset --model LAGUNITA --epoch 50 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name LALAGAN_2025-10-08
python test.py --dataroot DendriticFActinDataset --model LAGUNITA --epoch 100 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name LALAGAN_2025-10-08
python test.py --dataroot DendriticFActinDataset --model LAGUNITA --epoch 300 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name LALAGAN_2025-10-08
python test.py --dataroot DendriticFActinDataset --model LAGUNITA --epoch 400 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name LALAGAN_2025-10-08
python test.py --dataroot DendriticFActinDataset --model LAGUNITA --epoch 500 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name LALAGAN_2025-10-08
