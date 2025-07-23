# Chapter 7.4 - **How are normalization functions accelerated on NPU**

[TOC]



## Batchnorm

Batchnorm is typically used during training to normalize input features in a *minibatch* to a common range. This makes training more stable and more robust. Given x inputs in a minibatch of size N, we calculate mean and variance

$\mu = \Sigma(x_i)$/N

$\sigma^2 = \Sigma(x_i - \mu)^2$/N

Then we normalize all inputs, as follows

$x_i = (x_i - \mu)/(\sigma^2 + \epsilon)$



During inference, batchsize = 1 and therefore it is not possible to normalize using new data. A common practice is to use $\mu$ and $\sigma$ from training and merge it in the parameters of the previous layer.   



## Layernorm

Conceptually very similar to batchnorm, but one critical difference is 



- Batchnorm normalizes within a batch of samples, 
- Layernorm normalizes all features within one sample



This is a subtle, but a critical difference. We were able to skip batchnorm in inference assuming that use $\mu$ and $\sigma$ parameters are fused with previous layer parameters. But we do not have that freedom for Layernorm.



One additional challenge with Layernorm is the difficulty of quantization because of math operators sqrt and division.



## Softmax

$softmax(z_i) = (e^{z_i})/(\Sigma_i(e^{z_i}))$

Converts the logits into probabilities.

