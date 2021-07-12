# TA-GAN
### Task-Assisted Generative Adversarial Network for Resolution Enhancement and Modality Translation in Fluorescence Microscopy

This repository contains all code required to train and test our super-resolution microscopy image generation algorithm. Sample images and trained weights are included to test the method. The datasets can be downloaded at https://s3.valeria.science/flclab-tagan/index.html. The confocal and STimulated Emission Depletion (STED) microscopy images provided in the dataset were obtained from fixed and live primary cultures of rat hippocampal neurons and acquired on an Abberior Expert Line STED microscope. 

The code is based on conditional generative adversarial networks for image-to-image translation (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

### Installation

Clone this repository, then move to its directory:

```
git clone https://github.com/FLClab/TA-GAN.git
cd TA-GAN/
```

To make sure all prerequisites are installed, we advise to build and use the dockerfile included:

```
docker build TAGAN-Docker
nvidia-docker run -it --rm --user $(id -u) --shm-size=10g pytorch
```
If you are not familiar with Docker, you can also install build a virtual environment, activate it, and install all required packages using the requirements.txt file:
```
pip install virtualenv
virtualenv TA-GAN-venv 
source TA-GAN-venv/bin/activate (Linux / Mac OS)
TA-GAN-venv\Scripts\activate (Windows)
pip install -r requirements.txt --no-index
iip install torch==1.9.0
```

## TA-GAN for resolution enhancement

Different models are provided for specific use cases. The results presented in "Task-Assisted GAN for Resolution Enhancement and Modality Translation in Fluorescence Microscopy" were obtained using TA-GAN-axons (F-actin in axons of fixed cultured hippocampal neurons), TA-GAN-dendrites (F-actin in dendrites of fixed cultured hippocampal neurons), TA-GAN-synprot (synaptic proteins Bassoon, Homer-1c and PSD95 in fixed cultured hippocampal neurons) and TA-GAN-live (F-actin in axons and dendrites of living cultured hippocampal neurons).

#### Models

- TA-GAN-axons : TA-GAN model with binary segmentation used as the complementary task. 
- TA-GAN-dendrites : TA-GAN model with two classes semantic segmentation used as the complementary task.
- TA-GAN-synprot : TA-GAN model with two classes semantic segmentation used as the complementary task.
- TA-GAN-live : TA-GAN model with two classes semantic segmentation used as the complementary task. For this model, the segmentation network's weights are pre-trained and frozen.
- TA-GAN : Use this file to create your own TA-GAN model, specific to your datasets and analysis task of interest.

<img src="/figures/network.png">

Figure 1 : TA-GAN training architecture. Three networks are trained in parallel. The circles are the computed losses, with their specific color corresponding to the network they optimize. The segmentation serves as a complementary task compelling the generation of accurate structures, with the segmentation network optimized solely by the segmentation loss of the real STED. The discriminator classifies the input STED as either real or synthetic, conditional to its confocal. The generator is trained by the GAN loss, i.e its ability to fool the discriminator, by a pixel-wise loss comparing the synthetic and real STED, and by the segmentation loss of the synthetic STED.


#### Dataloaders

For training, make sure your dataset folder is split into 'train' and 'valid' subfolders. The data can be of various formats corresponding to the different
dataloaders provided. Custom dataloaders can easily be built using the template dataloader provided.

 - aligned_dataset : Tiff images with confocal and STED images concatenated along the channel axis.
 - mask_dataset : Tiff images with confocal, STED, and the segmentation labels concatenated along the channel axis. 
 - two_masks_dataset : Tiff images with confocal, STED, and two channels of segmentation labels, concatenated along the channel axis.
 - synprot_dataset : Tiff images with 6 channels ordered as [confocal_A, STED_A, confocal_B, STED_B, segmentation_A, segmentation_B]. Note that the confocal images were acquired with bigger pixels than the corresponding STED images (60 nm vs. 15 nm); to allow concatenation along the channel axis, the confocal images are upsampled by a factor of 4 with nearest-neighbor interpolation.
 - live_dataset : Tiff images, with confocal and STED images concatenated along the channel axis. This dataloader concatenates to the input modality (confocal) regions selected from the output modality (STED), along with a binary decision map indicating which regions from the output modality are given to the network. The generator should therefore take three channels as input. Before training, the user should decide the size of these regions by defining the variable *px* (line 66) and the random distribution from which the number of regions *n* is drawn (line 70).

<img src="/figures/dataset_modes.png">

## Reproducing the published results

### Training

Everything needed to reproduce the results published in "Task-Assisted Generative Adversarial Network for Resolution Enhancement and Modality Translation in Fluorescence Microscopy" is made available. The datasets can be downloaded here: https://s3.valeria.science/flclab-tagan/index.html. After downloading the datasets, run the following lines to train the model on one of the datasets provided. Note that the optimal hyperparameters are defined as default values for each model. **If you don't have access to a gpu, add the parameter ```gpu_ids=-1```.** 

**Axonal F-actin rings**
```
python3 train.py --dataroot=AxonalRingsDataset --model=TAGAN_AxonalRings
```
**Dendritic F-actin rings and fibers**
```
python3 train.py --dataroot=DendriticFActinDataset --model=TAGAN_Dendrites
```
**Synaptic Proteins**
```
python3 train.py --dataroot=SynapticProteinsDataset --model=TAGAN_Synprot
```
**Live F-actin** (with pretrained segmentation network)
You first need to download the trained segmentation network for F-actin in live-cell images and save it as checkpoints/LiveFActin/pretrained_net_S.pth
```
python3 train.py --dataroot=LiveFActinDataset --model=TAGAN_live --dataset_mode=live_train --continue --epoch=pretrained --name=LiveFActin
```
 
### Testing

The following lines can be directly used to test with the provided example data and the trained models.

**Axonal F-actin rings**
```
python3 test.py --dataroot=AxonalRingsDataset --model=TAGAN_AxonalRings --epoch=1000 --name=AxonalRings
```
<img src="/figures/axons_test.png">

**Dendritic F-actin rings and fibers**
```
python3 test.py --dataroot=DendriticFActinDataset --model=TAGAN_Dendrites --epoch=500 --name=DendriticFActin
```
<img src="/figures/dendrites_test.png">

**Synaptic Proteins**
```
python3 test.py --dataroot=SynapticProteinsDataset --model=TAGAN_Synprot --epoch=1000 --name=SynapticProteins
```
<img src="/figures/synprot_test.png">

**Live F-actin**
```
python3 test.py --dataroot=LiveFActinDataset --model=TAGAN_live --epoch=5000 --name=LiveFActin --phase=20201130_cs4 --dataset_mode=live_test --preprocess=center --crop_size=256
```

To test on your own images, create a folder and add the images to a subfolder inside. Use the parameters ```dataroot=folder_name``` and ```phase=subfolder_name``` to specify where the images are. Make sure the order of the channels and the pixel size corresponds to what the model has been trained with, i.e. use the same dataloader and model for training and testing.


### TA-GAN for modality translation: fixed-cell imaging to live-cell imaging

The TA-GAN architecture can also be used to translate imaging modalities while preserving the content relevant to the biological interpretation of the images. In our brief communication, this is used to translate fixed-cell images into live-cell images. 

(1) Train the modality translation TA-GAN model by downloading the dataset 'fixed_live' (https://s3.valeria.science/flclab-tagan/index.html) and running the following line:
```
python3 train.py --dataroot=fixed_live --model=TA-GAN-cycle --dataset_mode=fixed_live 
```
(2) Once trained, you can convert fixed-cell images into live-cell images:
```
python3 test.py --dataroot=fixed_live --model=TA-GAN-cycle --dataset_mode=fixed_live 
```
The generated images, along with the segmentation labels from the fixed-cell images, can then be used to train a segmentation network for live cells. The translated (fixed -> live) images can also be directly downloaded : https://s3.valeria.science/flclab-tagan/index.html. 

(3) Once downloaded (or generated using steps 1 and 2), use the following line to train the segmentation network for live cells:
```
python3 train.py --dataroot=translated_live --model=segmentation --dataset_mode=two_segmentation
```
(4) The segmentation network trained on the translated live-cell images is used to train the TA-GAN model. Copy-paste the trained segmentation model from step 3 to checkpoints/TA-GAN-live (or use the one that is already provided from the Github repository), and run the following line:
```
python3 train.py --dataroot=live --model=TA-GAN-live --dataset_mode=live --continue --epoch=pretrained
```
(5) Finally, test the generation of live-cell STED images from confocal images using the following line:
```
python3 test.py --dataroot=live --model=TA-GAN-live --dataset_mode=live --epoch=5000 --phase=20201130_cs4_ROI2
```
You can change the ```--phase``` parameter for any subfolder in the dataset titled "live" to test on different neurons. 

<img src="/figures/20201130_cs4_ROI2_conf.gif" width="250" height="250"/>  <img src="/figures/20201130_cs4_ROI2_fake.gif" width="250" height="250"/>\
<img src="/figures/20201130_cs4_ROI1_conf.gif" width="250" height="250"/>  <img src="/figures/20201130_cs4_ROI1_fake.gif" width="250" height="250"/>
