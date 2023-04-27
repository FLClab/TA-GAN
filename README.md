# TA-GAN
### Resolution Enhancement with a Task-Assisted GAN to Guide Optical Nanoscopy Image Analysis and Acquisition

This repository contains all code required to train and test the super-resolution microscopy image generation algorithm. Sample images and trained weights are included to test the method. The datasets and trained models can be downloaded at https://s3.valeria.science/flclab-tagan/index.html.

The [figures](/figures) folder contains the raw data and code to generate all figures from the paper. The images and scripts that were used to get the raw data can be provided upon request, but were too large to be included in this repository.

The code is based on conditional generative adversarial networks for image-to-image translation (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

- [System Requirements](#system)
- [Installation](#installation)
- [Documentation](#documentation)
- [Citation](#citation)

<a id="system"></a>
# System requirements
## Hardware requirements
Inference can be run on a computer without a GPU in reasonable time, requiring only that the RAM if sufficient for the size of the model and the loaded image. Use the parameter ```gpu-ids=-1``` if the computer has no GPU, ```batch_size=1``` to avoid filling the RAM, and ```crop_size``` to run the inference on smaller crops if the available RAM is insufficient for the complete image. Approximate inference times and RAM requirements are mentionned for each experiment in the section [Documentation/Testing](#testing).

For training TA-GAN, a GPU is necessary to reach reasonable computing times. Approximate training times and RAM requirements are mentionned for each experiment in the section [Documentation/Training](#training).

## Software requirements
### OS requirements
The source code was tested on Ubuntu 16.04, Ubuntu 20.04 and CentOS Linux 7.5.1804.

### Python dependencies
The source code was tested on Python 3.7. All required libraries are listed in the 'requirements.txt' file.

<a id="installation"></a>
# Installation
Clone this repository, then move to its directory:

```
git clone https://github.com/FLClab/TA-GAN.git
cd TA-GAN/
```

To make sure all prerequisites are installed, we advise to build and use the dockerfile included:

```
docker build -t tagan TAGAN-Docker
docker run -it --gpus all --rm --user $(id -u) --shm-size=10g -v /:/workspace/ tagan
```
Building the docker image requires 4.06 GB of free memory and takes around 10 minutes with a reliable Internet connection. The docker container was tested on Docker versions 20.10.12 and 18.06.0. For Docker version 18.06.0 and older, you may need to change the ```docker run``` command to ```nvidia-docker run``` to access a GPU inside the container.

If you are not familiar with Docker, you can also build a virtual environment, activate it, and install all required packages using the requirements.txt file:
```
pip install virtualenv
virtualenv TA-GAN-venv 
source TA-GAN-venv/bin/activate (Linux / Mac OS)
TA-GAN-venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

<a id="installation"></a>
# Documentation

## Pseudocode for the training algorithm
        opt, opt_val                      # Training options, validation options
        dataset = create_dataset(opt)     # Create a dataset given training options
        dataval = create_dataset(opt_val) # Create a dataset given validation options
        model = create_model(opt)         # Create a model given training options
        while (stopping criterion not satisfied)
            for data in dataset           # Iterate over all batches of data
                model.set_input(data)     # Unpack batch of data from dataset and apply preprocessing
                model.calculate_loss()    # Compute loss functions
                model.get_gradients()     # Get gradients
                model.optimize_parameters() # Update network weights
            update_learning_rate()        # Decrease learning rate if decaying lr is selected in options
        

## TA-GAN for resolution enhancement

Different models are provided for specific use cases. The results presented in "Task-Assisted GAN for Resolution Enhancement and Modality Translation in Fluorescence Microscopy" were obtained using TA-GAN-axons (F-actin in axons of fixed cultured hippocampal neurons), TA-GAN-dendrites (F-actin in dendrites of fixed cultured hippocampal neurons), TA-GAN-synprot (synaptic proteins Bassoon, Homer-1c and PSD95 in fixed cultured hippocampal neurons) and TA-GAN-live (F-actin in axons and dendrites of living cultured hippocampal neurons).

#### Models

- TA-GAN-axons : TA-GAN model with binary segmentation used as the complementary task. 
- TA-GAN-dendrites : TA-GAN model with two classes semantic segmentation used as the complementary task.
- TA-GAN-synprot : TA-GAN model with two classes semantic segmentation used as the complementary task.
- TA-GAN-live : TA-GAN model with two classes semantic segmentation used as the complementary task. For this model, the segmentation network's weights are pre-trained and frozen.
- TA-GAN : Use this file to create your own TA-GAN model, specific to your datasets and analysis task of interest.

<img src="/figures/README/network.png" width=500>

Figure 1 : TA-GAN training architecture. Three networks are trained in parallel. The circles are the computed losses, with their specific color corresponding to the network they optimize. The segmentation serves as a complementary task compelling the generation of accurate structures, with the segmentation network optimized solely by the segmentation loss of the real STED. The discriminator classifies the input STED as either real or synthetic, conditional to its confocal. The generator is trained by the GAN loss, i.e its ability to fool the discriminator, by a pixel-wise loss comparing the synthetic and real STED, and by the segmentation loss of the synthetic STED.


#### Dataloaders

For training, make sure your dataset folder is split into 'train' and 'valid' subfolders. The data can be of various formats corresponding to the different
dataloaders provided. Custom dataloaders can easily be built using the template dataloader provided.

 - aligned_dataset : Tiff images with confocal and STED images concatenated along the channel axis.
 - mask_dataset : Tiff images with confocal, STED, and the segmentation labels concatenated along the channel axis. 
 - two_masks_dataset : Tiff images with confocal, STED, and two channels of segmentation labels, concatenated along the channel axis.
 - synprot_dataset : Tiff images with 6 channels ordered as [confocal_A, STED_A, confocal_B, STED_B, segmentation_A, segmentation_B]. Note that the confocal images were acquired with bigger pixels than the corresponding STED images (60 nm vs. 15 nm); to allow concatenation along the channel axis, the confocal images are upsampled by a factor of 4 with nearest-neighbor interpolation.
 - live_dataset : Tiff images, with confocal and STED images concatenated along the channel axis. This dataloader concatenates to the input modality (confocal) regions selected from the output modality (STED), along with a binary decision map indicating which regions from the output modality are given to the network. The generator should therefore take three channels as input. Before training, the user should decide the size of these regions by defining the variable *px* (line 66) and the random distribution from which the number of regions *n* is drawn (line 70).

<img src="/figures/README/dataset_modes.png">

## Training and testing on your own images

**A new model MUST be trained for every change in biological context or imaging parameter. You can not apply the models we provide on your images if their acquisition parameters differ in any way.** We strongly believe the method we introduce is applicable to any context, if trained properly. To train a model on a new set of images, you need the following:
- An extensive set of images that covers everything you could expect to see in your test images. The model won't learn to generate structures it has never encountered in training. 
- Segmentation annotations for all the training images
- Computational resources (GPU)
- A model adapted to your specific use case. The general TA-GAN model can be used here, but you need to specify the number of input and output channels for the generator and the segmentation network to fit your use case (default = 1).
- A dataloader adapted to your images. We provide a custom dataloader that is heavily commented so that you can easily modify it to fit your needs.

When everything is ready, run the training:
```
python3 train.py --dataroot=<DATASET> --model=<MODEL> --dataset_mode=<DATALOADER> 
```
The default hyperparameters might not lead to the best results. You should play with the following hyperparameters:
- ```--niter``` (number of iterations)
- ```--niter_decay``` (number of iterations with decreasing learning rate)
- ```--batch_size``` (rule of thumb, use the largest batch size that fits on your GPU)
- ```--netG``` (architecture of the generator)
- ```--netS``` (architecture of the segmentation network)

The description and default values for all hyperparameters can be consulted in options/base_options, train_options and test_options.

## Reproducing the published results

<a id="training"></a>
### Training

Everything needed to reproduce the results published in "Task-Assisted Generative Adversarial Network for Resolution Enhancement and Modality Translation in Fluorescence Microscopy" is made available. The datasets can be downloaded here: https://s3.valeria.science/flclab-tagan/index.html. After downloading the datasets, run the following lines to train the model on one of the datasets provided. Note that the optimal hyperparameters are defined as default values for each model. **If you don't have access to a gpu, add the parameter ```--gpu_ids=-1```. If the only version of Python installed on your system is 3.x, use ```python``` instead of ```python3```.** Training times were computed on a GeForce RTX 2080 GPU.

**Axonal F-actin** (RAM required with default parameters: 3680 MiB; Training time with default parameters: 2 hours (7 seconds/epoch).)
```
python3 train.py --dataroot=AxonalRingsDataset --model=TAGAN_AxonalRings
```
**Dendritic F-actin** (RAM required with default parameters: 10642 MiB; Training time with default parameters: 9 hours (67 seconds/epoch).)
```
python3 train.py --dataroot=DendriticFActinDataset --model=TAGAN_Dendrites
```
**Synaptic Proteins** (RAM required with default parameters: 10606 MiB; Training time with default parameters: 8 hours (29 seconds/epoch).) 
```
python3 train.py --dataroot=SynapticProteinsDataset --model=TAGAN_Synprot
```
**S. Aureus cells**
```
python3 train.py --dataroot=SAureusDataset --model=TAGAN_SAureus
```
**Simulated nanodomains**
```
python3 train.py --dataroot=SimulatedNanodomainsDataset --model=TAGAN_SimulatedNanomdomains
```

**Live F-actin** (with pretrained segmentation network; RAM required with default parameters: 9690 MiB; Training time with default parameters: 33 hours (24 seconds/epoch).)
First download the trained segmentation network for F-actin in live-cell STED images [here](https://s3.valeria.science/flclab-tagan/index.html) and save it as checkpoints/LiveFActin/pretrained_net_S.pth
```
python3 train.py --dataroot=LiveFActinDataset --model=TAGAN_live --dataset_mode=live_train --continue --epoch=pretrained --name=LiveFActin
```
 
<a id="testing"></a>
### Testing

The following lines can be directly used to test with the provided example data and the trained models.

**Axonal F-actin rings** (RAM required with default parameters: 958 MiB; Inference time for 52 test images: <5 seconds.)
```
python3 test.py --dataroot=AxonalRingsDataset --model=TAGAN_AxonalRings --epoch=1000 --name=AxonalRings
```
<img src="/figures/README/axons_test.png" width=500>

**Dendritic F-actin rings and fibers** (RAM required with default parameters: <5000 MiB; Inference time for 26 test images: <1 minute.)
```
python3 test.py --dataroot=DendriticFActinDataset --model=TAGAN_Dendrites --epoch=500 --name=DendriticFActin
```
<img src="/figures/README/dendrites_test.png" width=500>

**Synaptic Proteins** (RAM required with default parameters: 10861 MiB; Inference time for 23 test images: <5 minutes.)
The test images for the Synaptic Proteins dataset are too large to fit on the tested GPU. We added the options ```tophalf``` and ```bottomhalf``` to split the images in two halves as a preprocessing step. The two halves can then be recombined using the function *combine_bottomtop.py* in the /data folder, by first changing the values for the input and output folders.
```
python3 test.py --dataroot=SynapticProteinsDataset --model=TAGAN_Synprot --epoch=200 --name=SynapticProteins --preprocess=bottomhalf
python3 test.py --dataroot=SynapticProteinsDataset --model=TAGAN_Synprot --epoch=200 --name=SynapticProteins --preprocess=tophalf
python3 combine_bottomtop.py
```
<img src="/figures/README/synprot_test.png" width=500>

**S. Aureus cells**
```
python3 test.py --dataroot=SAureusDataset --model=TAGAN_SAureus --epoch=best --name=SAureus_precise
```

<img src="/figures/README/deepbacs_test.png" width=500>

**Simulated nanodomains**
```
python3 test.py --dataroot=SimulatedNanodomainsDataset --model=TAGAN_SimulatedNanodomains --epoch=400 --name=SimulatedNanodomains
```

<img src="/figures/README/nanodomains_test.png" width=500>

**Live F-actin** (RAM required with default parameters: 1402 MiB; Inference time on 3 sequences of 30 test images: <10 seconds.)
You first need to download the trained segmentation network for F-actin in live-cell images [here](https://s3.valeria.science/flclab-tagan/index.html) and save it as checkpoints/LiveFActin/5000_net_S.pth
```
python3 test.py --dataroot=LiveFActinDataset --model=TAGAN_live --epoch=5000 --name=LiveFActin --phase=20201130_cs4 --dataset_mode=live_test --preprocess=center --crop_size=512
```

To test on your own images, create a folder and add the images to a subfolder inside. Use the parameters ```dataroot=folder_name``` and ```phase=subfolder_name``` to specify where the images are. Make sure the order of the channels and the pixel size corresponds to what the model has been trained with, i.e. use the same dataloader and model for training and testing.


## TA-GAN for domain adaptation: fixed-STED imaging to live-STED imaging

The TA-GAN architecture can also be used to translate imaging modalities while preserving the content relevant to the biological interpretation of the images. In our brief communication, this is used to translate fixed-cell images into live-cell images. 

(1) Train the modality translation TA-GAN model by downloading the dataset 'fixed_live' (https://s3.valeria.science/flclab-tagan/index.html) and running the following line:
```
python3 train_cycle.py --dataroot=FixedLiveDataset --model=TAGAN_cycle --dataset_mode=fixed_live 
```
(2) Once trained, you can convert fixed-cell images into live-cell images:
```
python3 test.py --dataroot=FixedLiveDataset --model=TAGAN_cycle --dataset_mode=fixed_live --phase=train
```
The generated images, along with the segmentation labels from the fixed-cell images, can then be used to train a segmentation network for live cells. The translated (fixed -> live) images can also be directly downloaded : https://s3.valeria.science/flclab-tagan/index.html. 

(3) Once downloaded (or generated using steps 1 and 2), use the following line to train the segmentation network for live cells:
```
python3 train.py --dataroot=DomainAdaptedLiveDataset --model=segmentation
```
(4) The segmentation network trained on the translated live-cell images is used to train the TA-GAN model. Copy-paste the trained segmentation model from step 3 to checkpoints/TA-GAN-live/net_S_pretrained.pth (or use the one that is already provided from the Github repository), and run the following line:
```
python3 train.py --dataroot=LiveFActinDataset --model=TAGAN_live --dataset_mode=live_train --continue --epoch=pretrained --name=LiveFActin
```
(5) Finally, test the generation of live-cell STED images from confocal images using the following line:
```
python3 test.py --dataroot=LiveFActinDataset --model=TAGAN_live --dataset_mode=live_test --epoch=5000 --phase=20201130_cs4 --name=LiveFActin
```
You can change the ```--phase``` parameter for any subfolder in the dataset titled "live" to test on different regions and neurons. 

<img src="/figures/README/20201130_cs4_ROI2_conf.gif" width="250" height="250"/>  <img src="/figures/README/20201130_cs4_ROI2_fake.gif" width="250" height="250"/>\
<img src="/figures/README/20201130_cs4_ROI1_conf.gif" width="250" height="250"/>  <img src="/figures/README/20201130_cs4_ROI1_fake.gif" width="250" height="250"/>

## Baselines
### For resolution enhancement / denoising
- DNCNN : https://github.com/yinhaoz/denoising-fluorescence
- CARE : https://github.com/CSBDeep/CSBDeep
- 3D-RCAN : https://github.com/AiviaCommunity/3D-RCAN
- Pix2Pix : Use the parameter ```--model=pix2pix``` with a paired dataset (low- and high-resolution)

<img src="/figures/README/baselines_v0.png" width=500px>

### For modality translation
- Pix2Pix (paired) : Use the parameter ```--model=pix2pix``` with a paired dataset (different modalities)
- Cycle-GAN (unpaired) : Use the parameter ```--model=cycle-gan``` with an unpaired dataset (different modalities)

# Citation
If using, please cite the following paper :
Bouchard, C., Wiesner, T., Deschênes, A., Bilodeau, A., Lavoie-Cardinal, F., & Gagné, C. (2021). Resolution Enhancement with a Task-Assisted GAN to Guide Optical Nanoscopy Image Analysis and Acquisition. *bioRxiv*.

```
@article{bouchard2022resolution,
  title={Resolution Enhancement with a Task-Assisted GAN to Guide Optical Nanoscopy Image Analysis and Acquisition},
  author={Bouchard, Catherine and Wiesner, Theresa and Desch{\^e}nes, Andr{\'e}anne and Bilodeau, Anthony and Turcotte, Benoît and Gagn{\'e}, Christian and Lavoie-Cardinal, Flavie},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}

```
