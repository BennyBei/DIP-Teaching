# Assignment 3 - Play with GANs

### This is Bei's impletation of DIP assignment3

### Resources:
- [DragGAN](https://vcai.mpi-inf.mpg.de/projects/DragGAN/): [Implementaion 1](https://github.com/XingangPan/DragGAN) & [Implementaion 2](https://github.com/OpenGVLab/DragGAN)
- [Facial Landmarks Detection](https://github.com/1adrianb/face-alignment)

---
## Requirements

### Pix2Pix with GAN
Based on enviroment and dataset used in homework2.

To run training

```train
python train.py
```

Noted, one should adjust batch size in line 142, 143 according to VRAM, otherwise the training will be extremely slow.

## Results

#### train results
<center>
    <img src = "./Pix2Pix_GAN/train_results/epoch_795/result_1.png"
        width = "80%">
</center>
<center>
    <img src = "./Pix2Pix_GAN/train_results/epoch_795/result_2.png"
        width = "80%">
</center>
<center>
    <img src = "./Pix2Pix_GAN/train_results/epoch_795/result_5.png"
        width = "80%">
</center>

#### val results
<center>
    <img src = "./Pix2Pix_GAN/val_results/epoch_795/result_1.png"
        width = "80%">
</center>
<center>
    <img src = "./Pix2Pix_GAN/val_results/epoch_795/result_2.png"
        width = "80%">
</center>
<center>
    <img src = "./Pix2Pix_GAN/val_results/epoch_795/result_5.png"
        width = "80%">
</center>

GAN gets better result than FCN_network.