# Fundus-Enhanced Disease-Aware Distillation Model for Retinal Disease Classification from OCT Images

## Introduction

This is the pytorch implementation for the paper "[Fundus-Enhanced Disease-Aware Distillation Model for Retinal Disease Classification from OCT Images](https://arxiv.org/abs/2308.00291)"

![image-20230619155720859](figure/framework.png)

## Data Preparation

- Dataset

  We collect a new dataset TOPCON-MM with fundus and OCT images in Guangdong Provincial Hospital of Integrated Traditional Chinese and Western Medicine using a Topcon Triton swept-source OCT featuring multimodal fundus imaging. We have separated our in-house dataset  at the patient level, maintained a training-to-test set ratio of approximately 8:2. Click [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/lwangdk_connect_ust_hk/Eey6EX7oETNBq9Im8XFFfYIBernNJaTOSzS01GO3-ybyew?e=AFsAnx) to download the dataset.

- Data Preprocessing

  For fundus images, we used contrast-limited adaptive histogram equalization to improve image quality. For OCT images, we adopted 3x3 median filter to reduce the background noise. We utilized data augmentation including random crop, flip, rotation, and changes in contrast, saturation, and brightness. All the images are resized to 448×448 before feeding into the network.

- Structure of data folder

  ```
  image_data/
  └── topcon-mm/
      ├── train/
      │   ├── cfp.txt
      │   ├── oct.txt
      │   └── Images/
      │       ├── fundus-images/
      │       └── oct-images/
      ├── val/
      │   ├── cfp.txt
      │   └── oct.txt
      └── test/
          ├── cfp.txt
          └── oct.txt
  ```

## Implementation

1. Check dependencies

   ```
   matplotlib==3.5.3
   numpy==1.21.5
   opencv-python==4.7.0.68
   Pillow==9.3.0
   python==3.7.10
   scikit-learn==0.24.2
   scipy==1.7.3
   torch==1.11.0
   torchcam==0.3.2
   torchvision==0.12.0
   ```

2. Train model

   First, we pretrained the fundus single-modal model with fundus images.

   ```
   python train_fundus.py --train_collection 'image_data/topcon-mm/train' \
                   --val_collection 'image_data/topcon-mm/val' \
                   --test_collection 'image_data/topcon-mm/test' \
                   --model_configs 'config_fundus.py' \
   ```

   Next, the fundus-enhanced model for OCT images is trained.

   ```
   python train.py --train_collection 'image_data/topcon-mm/train' \
                   --val_collection 'image_data/topcon-mm/val' \
                   --test_collection 'image_data/topcon-mm/test' \
                   --model_configs 'config.py' \
                   --alpha 2 \
                   --temperature 4 \
                   --beta 1 \
                   --batch_size 8 \
                   --checkpoint $pretrained_model_for_fundus
   ```

## Citation
If this code is helpful for you, please cite:
```
@inproceedings{wang2023fundus,
  title={Fundus-Enhanced Disease-Aware Distillation Model for Retinal Disease Classification from OCT Images},
  author={Wang, Lehan and Dai, Weihang and Jin, Mei and Ou, Chubin and Li, Xiaomeng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={639--648},
  year={2023},
  organization={Springer}
}
```

## Contact
Lehan Wang (lwangdk@connect.ust.hk), 
Xiaomeng Li (eexmli@ust.hk)
