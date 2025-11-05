#!/bin/bash 

# python test.py --dataroot AxonalRingsDataset --model SAGAN --epoch 100 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name SAGANAxonalFactin_2025-11-05
python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model SAGAN --epoch 150 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name SAGANAxonalFactin_2025-11-05
# python test.py --dataroot AxonalRingsDataset --model SAGAN --epoch 200 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name SAGANAxonalFactin_2025-11-05
python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model SAGAN --epoch 250 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name SAGANAxonalFactin_2025-11-05
# python test.py --dataroot AxonalRingsDataset --model SAGAN --epoch 300 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name SAGANAxonalFactin_2025-11-05
# python test.py --dataroot AxonalRingsDataset --model SAGAN --epoch 400 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name SAGANAxonalFactin_2025-11-05


# python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model TAGAN_AxonalRings --epoch 100 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name TAGANAxonalFactin_2025-11-04
python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model TAGAN_AxonalRings --epoch 150 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name TAGANAxonalFactin_2025-11-04
# python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model TAGAN_AxonalRings --epoch 200 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name TAGANAxonalFactin_2025-11-04
python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model TAGAN_AxonalRings --epoch 250 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name TAGANAxonalFactin_2025-11-04
# python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model TAGAN_AxonalRings --epoch 300 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name TAGANAxonalFactin_2025-11-04
# python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model TAGAN_AxonalRings --epoch 400 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name TAGANAxonalFactin_2025-11-04

# python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model pix2pix --epoch 100 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name Pix2PixAxonalFactin_2025-11-04
python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model pix2pix --epoch 150 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name Pix2PixAxonalFactin_2025-11-04
# python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model pix2pix --epoch 200 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name Pix2PixAxonalFactin_2025-11-04
python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model pix2pix --epoch 250 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name Pix2PixAxonalFactin_2025-11-04
# python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model pix2pix --epoch 300 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name Pix2PixAxonalFactin_2025-11-04
# python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model pix2pix --epoch 400 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name Pix2PixAxonalFactin_2025-11-04
# python test.py --dataroot AxonalRingsDataset --input_nc 1 --output_nc 1 --model pix2pix --epoch 500 --checkpoints_dir /home-local/Frederic/baselines/SR-baselines --name Pix2PixAxonalFactin_2025-11-04