# TA-GAN
## Task-Assisted Generative Adversarial Network for Resolution Enhancement and Modality Translation in Fluorescence Microscopy

### TA-GAN for resolution enhancement in fixed-cell imaging

<img src="/figures/network.png">

Three resolution enhancement models, trained on three different datasets of fixed-cell images, are provided:

- F-actin in axons

<img src="/figures/axons_test.png" width="50%" height="50%">

To test with the provided images:

```
python test.py --dataroot=actin --dataset_mode=masks --model=pix2pix_seg --preprocess=none --name=actin --epoch=1000

```

- F-actin in dendrites

<img src="/figures/dendrites_test.png" width="50%" height="50%">

To test with the provided images:

```
python test.py --dataroot=actin --dataset_mode=masks --model=pix2pix_seg --preprocess=none --name=actin --epoch=1000
```

- Synaptic proteins clusters

<img src="/figures/synprot_test.png" width="50%" height="50%">

To test with the provided images:

```
python test.py --dataroot=actin --dataset_mode=masks --model=pix2pix_seg --preprocess=none --name=actin --epoch=1000
```

### TA-GAN for modality translation: fixed-cell imaging to live-cell imaging


### TA-GAN for resolution enhancement in live-cell imaging

<img src="/figures/20201130_cs4_ROI2_conf.gif" width="250" height="250"/>  <img src="/figures/20201130_cs4_ROI2_fake.gif" width="250" height="250"/>\
<img src="/figures/20201130_cs4_ROI1_conf.gif" width="250" height="250"/>  <img src="/figures/20201130_cs4_ROI1_fake.gif" width="250" height="250"/>
