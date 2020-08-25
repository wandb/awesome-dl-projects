# Modern Data Augmentation Techniques for Computer Vision

## About

Implementation of various data augmentation techniques in TensorFlow 2.x. They can be easily used in your training pipeline. This repository contain the supplementary notebooks for the [Modern Data Augmentation Techniques for Computer Vision](https://app.wandb.ai/authors/tfaugmentation/reports/Modern-Data-Augmentation-Techniques-for-Computer-Vision--VmlldzoxNDA2NTc)(Weights and Biases) report.

## Techniques Covered

* [Cutout](https://github.com/ayulockin/DataAugmentationTF/blob/master/CIFAR_10_with_Cutout_Augmentation.ipynb)
* [Mixup](https://github.com/ayulockin/DataAugmentationTF/blob/master/CIFAR_10_with_Mixup_Augmentation.ipynb)
* [CutMix](https://github.com/ayulockin/DataAugmentationTF/blob/master/CIFAR_10_with_CutMix_Augmentation.ipynb)
* [Augmix](https://github.com/ayulockin/DataAugmentationTF/blob/master/Cifar_10_with_AugMix_Augmentation.ipynb)

**Note**: Cutout, Mixup and CutMix are implememted in `tf.data` and can be found in the linked colab notebooks. I am using TensorFlow 2.x implementation of AugMix by [Aakash Nain](https://twitter.com/A_K_Nain?s=09). His repo can be found [here](https://github.com/AakashKumarNain/AugMix_TF2). The [fork](https://github.com/ayulockin/AugMix_TF2) of this repo contains Weights and Biases integration and some additional command like arguments for more control.   

## Result

Check out the linked [report](https://app.wandb.ai/authors/tfaugmentation/reports/Modern-Data-Augmentation-Techniques-for-Computer-Vision--VmlldzoxNDA2NTc) for:

* The comparative study of these augmentation techniques. 
* Augmentation implementations.
* Evaluation of these augmentation techniques against [Cifar-10-C dataset](https://zenodo.org/record/2535967).

## Model Used

![ResNet-20](https://github.com/ayulockin/DataAugmentationTF/blob/master/images/model.png)

