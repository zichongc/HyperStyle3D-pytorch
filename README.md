# HyperStyle3D-pytorch
An unofficial implementation of HyperStyle3D: Text-Guided 3D Portrait Stylization via Hypernetworks (Chen et al., arxiv 2023) [[arxiv](https://arxiv.org/abs/2304.09463)]

<div align="center">
<img src=./assets/demo.png>
</div>


## Setup

### Preparison of pretrained model
* Download StyleSDF model pretrained on FFHQ `ffhq1024x1024.pt` (from[ here](None)) and move to `./StyleSDF/full_models`
* Download ArcFace pretrained model `model_ir_se50.pth` (from[ here](None)) and move to `./util/pretrained_models`. 
* Download CLIP pretrained model `ViT-B-32.pt` (from[ here](None)) and move to `./util/pretrained_models`.  
* (Optional) Download a pretrained matting model `MODEL.pth` (from[ here](None)) and move to `./util/pretrained_models`. 


## Demo 

## Acknowledgments
This code is inspired by rosinality's [StyleSDF](https://github.com/royorel/StyleSDF).