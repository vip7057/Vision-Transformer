# Vision-Transformer (Visual Transformers)
><center><img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png" alt="Alternative text"/></center>)
><center><figcaption>Fig 1. Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale."https://arxiv.org/pdf/2010.11929.pdf. </figcaption></center>                 


Transformers have been studied in the context of sequence-to-sequence modelling applications like natural language processing (NLP). Their superior performance to LSTM-based Recurrant neural network gained them a powerful reputation, thanks to their ability to model long sequences. A couple of years ago, transformers have been adapted to the [visual domain](https://arxiv.org/abs/2010.11929) and suprisingly demonstrated better performance compared to the long standing convolutional neural networks conditioned to large-scale datasets. Thanks to their ability to capture global semantic relationships in an image, unlike, CNNs which capture local information within the vicinty of the convolutional kernel window.

This repository implements the building blocks of visual transformers (LightViT). Afterwards, trains them on classification task using MNIST and Fashion-MNIST datasets.

---

## 1. Image Patches and Linear Mapping

### A) Image Patches
Transfomers were initially created to process sequential data. In case of images, a sequence can be created through extracting patches. To do so, a crop window should be used with a defined window height and width. The dimension of data is originally in the format of *(B,C,H,W)*, when transorfmed into patches and then flattened we get *(B, PxP, (HxC/P)x(WxC/P))*, where *B* is the batch size and *PxP* is total number of patches in an image. In this example, P=7. 


*Output*: A function that extracts image patches. The output format should have a shape of (B,49,16). The function will be used inside *LightViT* class.

### B) Linear Mapping

Afterwards, the input are mapped using a linear layer to an output with dimension *d* i.e. *(B, PxP, (HxC/P)x(WxC/P))* &rarr; *(B, PxP, d)*. The variable d can be freely chosen, however, we set here to 8. 

*Output*: A linear layer should be added inside *LightViT* class with the correct input and output dimensions, the output from the linear layer should have a dimension of (B,49,8). 

---
## 2. Insert Classifier Token and Positional embeddings

### A) Classifier Token
Beside the image patches, also known as tokens, an additional special token is appended to the the input to capture desired information about other tokens to learn the task at hand. Lateron, this token will be used as input to the classifier to determine the class of the input image. To add the token to the input is equivilant to concatentating a learnable parameter with a vector of the same dimension *d* to the image tokens. 

*Output* A randomly initialised learnable parameter to be implemented inside *LightViT* class. Used [PyTorch built-in function](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html) to create a PyTorch parameter.

### B) Positional Embedding

To preserve the context of an image, positional embeddings are associated with each image patch. Positional embeddings encodes the patch positions using sinusoidal waves, however, there are other techniques. We follow the definition of positional encoding in the original transformer paper of [Vaswani et. al](https://arxiv.org/abs/1706.03762), which sinusoidal waves. You'll be required to implement a function that creates embeddings for each coordinate of every image patch. 

---
## 3. Encoder Block

<center><img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" alt="Alternative text" width="400" height="500"/></center> 
<center><figcaption>Fig 2. Transformer Encoder."https://arxiv.org/pdf/2010.11929.pdf. </figcaption></center>  

This part implements the main elements of an encoder block. A single block contains layer normalization (LN), multi-head self-attention (MHSA), and a residual connection.  

### A) Layer Normalization
[Layer normailzation](https://arxiv.org/abs/1607.06450), similar to other techniques, normalizes an input across the layer dimension by subtracting mean and dividing by standard deviation. 

### B) MHSA
<center><img src="https://production-media.paperswithcode.com/methods/multi-head-attention_l1A3G7a.png" alt="Alternative text" width="300" height="400"/></center> 
<center><figcaption>Fig 2. Multi-Head Self Attention."https://arxiv.org/pdf/1706.03762v5.pdf. </figcaption></center>  
  
 The attention module derives an attention value by measuring similarity between one patch and the other patches. To this end, an image patch with dimension *d* is linearly mapped to three vectors; query **q**, key **k**, and value **v** , hence a distint linear layer should be instantiated to get each of the three vectors. To quantify attention for a single patch, first, the dot product is computed between its **q** and all of the **k** vectors and divide by the square root of the vector dimension i.e. *d* = 8. The result is passed through a softmax layer to get *attention features* and finally multiple with **v** vectors associated with each of the **k** vectors and sum up to get the result. This allows to get an attention vector for each patch by measuring its similarity with other patches.
 
  This process should be repeated **N** times on each of the **H** sub-vectors of the 8-dimensional patch, where **N** is the total number of attention blocks. In our case, let **N** = 2, hence, we have 2 sub-vectors, each of length 4. The first sub-vector is processed by the first head and the second sub-vector is process by the second head, each head has distinct Q,K, and V mapping functions of size 4x4. 
 

### C) Residual Connection
Residual connections (also know as skip connections) add the original input to the processed output by a network layer e.g. encoder. They have proven to be useful in deep neural networks as they mitigate problems like exploding / vanishing gradients. In transformer, the residual connection is adding the original input to the output from LN &rarr; MHSA. All of the previous operations could be implemented inside a seperate encoder class.

The last part of an encoder, is to a inser another residual connection between the input to the encoder and the output from the encoder passed through another layer of LN &rarr; MLP. The MLP consists of 2 layers with hidden size 4 times larger than *d*.

---
## 4. Classification Head
The final part of implemeting a transformer is adding a classification head to the model inside *LightViT* class. i.e. a linear layer that accepts input of dimension *d* and outputs logits with dimension set to the number of classes for the classification problem at hand.

---
## 5a. Model Training for MNIST
Checkout the ipynb file in the repository for complete implemetation in Pytorch.

---
## 5b. Model Training for FashionMNIST
Checkout the ipynb file in the repository for complete implemetation in Pytorch.
