# Chapter 7.2 - **How are Conv2d operations accelerated on NPU**

[TOC]



## **2D Convolutions**

2D convolution is the backbone of a CNN (Convolutional Neural Network). Convolutions have been used in image processing for a long time. Some examples are edge detection, noise filtering, blurring, sharpening etc. Mathematically speaking, a convolution is an integral that expresses the amount of overlap of one function f as it is shifted over another function g. In image processing and computer vision, convolutions have been used for various purposes like edge detection (Sobel filter), sharpen/blur images etc. In Machine Learning, convolutions are used to extract low level features from an input signal. This is done by sliding a set of filter kernels on the input tensor. The figure below is an example of a 3x3 kernel applied on a 3-channel RGB image.

![](C:\work\edgeai_book\Conv2d.JPG)

​											2D Convolution on an RGB Image

A CNN graph has several convolution layers, with each layer having several learnable kernels. After training, each kernel learns to detect some unique features in the image which can later be used for feature patching, pattern recognition and even content generation.

Mathematical formula for conv2d is
$$
S(i,j)  = (I*K)(i,j) = \sum_{i=1}^m  \sum_{j=1}^n I(i+m,j+n).K(m,n)
$$
Input ***I\*** is a 3D tensor of shape Wi x Hi x Ci

Kernel ***K\*** is a 4D tensor of size k x k x Ci x Co, assuming a square kernel of size k x k 

Output ***S\*** is a 3D tensor of size Wo x Ho x Co

With unity stride and no dilation, the following relationship applies

- Wo = Wi - k + pad - 1
- Ho = Hi – k + pad - 1

Convolution is the most computationally expensive operation in a neural net.

- One convolution filter takes k\*k \*Ci* multiplications and additions. Total *2\* k\*k \*Ci* operations per filter.
- There are *Co* filters in a layer.
- *Wo\*Ho\*Co* outputs need to be generated.
- *2\* k\*k \*Ci \* Co \* Wo\*Ho* operations for 1 convolution layer.

2D Convolution can be viewed as a 7-level nested for loop

`for n=0:batchsize`

  `for cout=0:Co`

​    `for cin=0:Ci`

​      `for h=0:Ho`

​        `for w=0:Wo`

​          `for k_v = 0:ky`

​            `for k_h = 0:kx`

​              `O[n][co][h][w] += kernel[co][ci][k_h][ k_w] * input[n][ci][ h + k_h][ w + k_w]]`



This needs to be repeated for all the convolution layers in a network. 

All of this needs to be repeated *ips* times in a second, where *ips* is the number of times a CNN graph is executed per second.

 

**Example** – Darknet-53 (640x640) requires 155 Billion operations for 1 inference. 

@30 ips, the number of computations needed is **4.65 Trillion operations per second**.

 Therefore, an efficient implementation of convolution on the hardware is critical to achieving high performance at low system cost and power.

 

Although, it is possible to run Conv2d on Vector (VPUs), in this chapter we will focus on Conv2d acceleration on a Matrix Processing Unit (MPU) 



### **Accelerating Conv2D on MPU**

#### **Im2col method**

Our least favorite method. Requires the input image (HxWxC) to be expanded by a factor of $k^2$ and reshaped to $H*W$ x   $k^2 * Ci$. Kernel tensor is in shape $k^2*Ci$ x $Co$. This results in a GeMM of form $C=A . B$

where

- $A$ is a matrix of shape MxN
- $B$ is a matrix of shape NxK
- $M = H*W$
- $K = k^2*Ci$
- $N = Co$

On a VPU, parallelization can be done along the M dimension or N dimension. In either case, this achieves 100% utilization when M (or N) is a multiple of VLMAX, (and of-course memory is not the bottleneck).

 

On an MPU, the two dimensions for parallelization chosen are M and N. This achieves 100% MAC array utilization when M is a multiple of MAC array width (TMMAX, in case of RISC-V matrix) and N is a multiple of MAC array height (TNMAX, in case of RISC-V matrix). For large tensors this is not difficult to achieve, though there are exceptions. 

 

If 100% PE utilization is achievable, where is the problem? The expansion step is where the problem lies. This has two major disadvantages

1. It increases memory footprint by a factor of **$k^2$**.

2. VPU and MPU are not designed for doing this expansion. Hence, it falls back to scalar and becomes a performance bottleneck.

   

There is a proposed **[implicit im2col](https://arxiv.org/abs/2110.03901)** method, but we leave to the reader to explore it.



#### **Conv2D using Outer Product**

We are all engineers. Why not jump right into the pseudocode.

```
 for i=0:Ho                           
    for j=0:Wo:T                 
        for cout=0:Co:T
            for k_v = 0:k
                 for k_h = 0:k
                       input_slice = input[i+y, j:j+T] #input vector of length T elements
                       kernel_slice = kernel[y, x, cout*T:(cout+1)T] #weight vector of length T elements
                       ACC += outer(input_slice, kernel_slice) #Vector - Vector Outer Product and accumulation
            out_tile[i,j] = ACC #Write TxT convolution result to output tensor
```

 T is the tile size of matrix MAC array 

The inner loop loads T elements from the input and T elements from weight matrix. The tensor layout assumed here is NHWC (as discussed in chapter 6), which means that  the input tensor is laid out "channel-wise". 





Important thing to note is that we have chosen the dimensions Wo and Co for parallelization. On an Outer-Product based MPU, this allows us to achieve **100% MAC array utilization** without any im2col like transformation, as long as

- Wo is a multiple of T

- Co is a multiple of T

  

 

### **Other types of 2D convolutions**

So far we have considered the simplest form of Conv2d. There are some variations which can potentially have a performance impact. Let’s look at some of them

 

#### **Pointwise Convolutions**

The only difference is that the kernel is of size 1x1xCi. Has no impact on performance.

 