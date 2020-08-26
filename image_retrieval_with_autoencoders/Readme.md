# [Towards representation learning for an image retrieval task](https://app.wandb.ai/authors/image-retrieval/reports/Towards-Representation-Learning-for-an-Image-Retrieval-Task--VmlldzoxOTY4MDI)

Authors:

1. [Aritra Roy Gosthipaty](https://app.wandb.ai/ariG23498)
2. [Souradip Chakraborty](https://app.wandb.ai/souradip)

## Introduction:

In this report, we approach the image retrieval problem from an `unsupervised perspective`. The foundation of our work lies in the latent space representation of the images learned through a self-supervised learning task. The goal here is to capture the latent space `embeddings` of images and then try to determine the `distance` among them in the latent space. With this approach, we are focusing on the perceptual realm of an image. We validate the quality of the learned representations through a Clustering task and measure its performance through the normalised mutual information score & rand index. Then we identify the issues in learning in a purely unsupervised scenario (link) and show the enhancement in the information content of the learned representations with a `hint of supervision`. We train a regularised auto encoder with the supervised information. We validate the performance in a retrieval framework for the test set.

## Contents:

1. `AutoEncoders_for_Image_Retrieval.ipynb`: This notebook houses the experiments done for image retrieval with a vanilla autoencoder and its latent space encodings.
2. `AutoEncoders_with_Supervision_for_Image_Retrieval.ipynb`: In this notebook the latent space of an autoencoder is provided a hint of supervision. The supervision helps in better clustering performance.

## Results:

|                 Models                 |  NMI  |  RI   |
| :------------------------------------: | :---: | :---: |
|          Vanilla Autoencoder           | 0.074 | 0.811 |
| Autoencoder with a hint of supervision | 0.433 | 0.876 |

## t-SNE of the embeddings:

1. **Vanilla Autoencoder**: The plot on the left shows the t-SNE of the embeddings. There are 10 colors each for the 10 classes. We can see that points of  the same colors do clutter together but not that much to be prominent.  The plot does not seem that bad so we go ahead and cluster and retrieve  images in the later phase.

![Auto.png](https://i.ibb.co/F66KdRj/t-sne-auto.png)

2. **Autoencoder with supervision**: The added information has been incorporated in the representation  learned by the image through the classification bottleneck which  together with the self-supervised loss helps the model to learn an  efficient representation of the latent space.

![Supervision.png](https://i.ibb.co/9rT9cjX/t-sne-auto-supervision.png)

## Reach the authors:

| Name                                                  | Twitter                                                 | Github                                                       |
| ----------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| [Aritra Roy Gosthipaty](https://arig23498.github.io/) | [@ariG23498](https://twitter.com/ariG23498/)            | [ariG23498](https://github.com/ariG23498/)                   |
| Souradip Chakraborty                                  | [@SOURADIPCHAKR18](https://twitter.com/SOURADIPCHAKR18) | [souradip-chakraborty](https://github.com/souradip-chakraborty) |