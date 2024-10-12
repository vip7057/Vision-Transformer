# Vision-Transformer (Visual Transformers)
>(<center><img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png" alt="Alternative text"/></center>)
><center><figcaption>Fig 1. Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale."https://arxiv.org/pdf/2010.11929.pdf. </figcaption></center>                 


Transformers have been studied in the context of sequence-to-sequence modelling applications like natural language processing (NLP). Their superior performance to LSTM-based Recurrant neural network gained them a powerful reputation, thanks to their ability to model long sequences. A couple of years ago, transformers have been adapted to the [visual domain](https://arxiv.org/abs/2010.11929) and suprisingly demonstrated better performance compared to the long standing convolutional neural networks conditioned to large-scale datasets. Thanks to their ability to capture global semantic relationships in an image, unlike, CNNs which capture local information within the vicinty of the convolutional kernel window.

This repository implements the building blocks of visual transformers (LightViT). Afterwards, trains them on classification task using MNIST and Fashion-MNIST datasets.

