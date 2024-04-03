# Performance-Comparison-on-PCAM-dataset

Self-Supervised Learning, SimCLR vs. Transfer Learning, pretrained weight (ImageNet) 

## Dataset
- [PCam dataset](https://github.com/basveeling/pcam)
- PCam dataset consists of 327,680 color images (96 x 96px) extracted from histopathologic scans of lymph node sections.
- Each image is annoted with a binary label indicating presence of metastatic tissue.
- PCam provides a new benchmark for machine learning models: bigger than CIFAR10, smaller than imagenet, trainable on a single GPU.

## Experimental Set-up List
1. Frozen, ImageNet weight 
2. Fine tuning, ImageNet weight 
3. Frozen, SimCLR
4. Fine tuning, SimCLR

## Backbone List 
1. VGG16 (pytorch implementation)
2. ResNet34 (pytorch implementation)


## More Information

requirements.txt
- for Windows (pip list ...)
- for CUDA version 12.4