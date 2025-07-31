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



```
 for n in range (batchsize):
   for cout in range(Co):
     for cin in range (Ci):
       for h in range (Ho):
         for w in range(Wo):
           for k_v in range(ky):
             for k_h in range(kx):
               O[n][co][h][w] += kernel[co][ci][k_h][ k_w] * input[n][ci][ h + k_h][ w + k_w]]           
         
```



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


#### **Conv2D using Dot Products**

We are all engineers. Why not jump right into the pseudocode.

```
def conv_nhwc(in_tensor, wt_tensor):
    N,H_in,W_in,C_in = in_tensor.shape
    k,_,_,C_out = wt_tensor.shape

    num_cycles = 0

    H_out = H_in-k+1                      #Height of output channel
    W_out = W_in-k+1                      #Width of output channel

    for n in range(N):
      for y in range(H_out):
        for i in range(k):
          for j in range(k):
            for c_out in range(0,C_out,T):                                      #parallelization dim2
              for x in range(0,W_out,T):                                        #parallelization dim1
		        for c_in in range(0,C_in,T):
		          in_tile = in_tensor[n,y+i,x+j:x+j+T,c_in:c_in+T]              #input tile of size TxT
		          wt_tile = wt_tensor[i,j,c_in:c_in+T,c_out:c_out+T]            #input tile of size TxT
		          out_tile, cycles_opa = tiled_dot_product(in_tile, wt_tile)    #output tile of size TxT
              out[n,y,x:x+Ty,c_out:c_out+Tx] += out_tile
            
```

T is the tile size of matrix MAC array, and is a hardware constant. 

N=batch size and can be assumed to be 1 for law latency edge AI inferencing.

 

Convolution is computed by dividing the input tensor and weight tensor into tiles as follows

- Input tile is of size T x T.  It uses the last two dimensions in NHWC layout i.e. W and C. If C_in <= T, each tile contains $T^2$ elements which are physically contiguous in memory. Hence, we can use unit-stride load without any drop in performance. What happens if C_in > T? Hold on to that question, we will come back to it.
- Weight tile is also of T x T. The two dimensions it uses are C_in and C_out. Here also If C_in <= T, each tile contains $T^2$ elements which are physically contiguous in memory.
- Output tile generated is also in NHWC layout. The last two dimensions are W and C. Therefore, if  C_out <= T the computed tile can be written back to memory with unit-stride.



This gives us a highly efficient 2D Convolution. On an Outer-Product based MPU (as discussed in chapter 7.1 ) , this allows us to achieve **100% MAC array utilization** without any im2col like transformation, as long as

- W_out is a multiple of T

- C_out is a multiple of T

  

if these rules are not followed, convolution still works and generates the correct output. However, the loop will have a tail which will work on small tiles. For the tail, MAC array utilization will be <100%. For e.g. if convolution is done on 224x224x3 image, W_out (after padding) = 224. If MAC array width is 64, Conv2d kernel processes along Width dimension in chunks of 64, 64, 64 and 32. For the last iteration of the loop, $utilization = 32/64 = 50\%$. This takes total utilization for the layer down to $224/(4*64) = 87.5\%$



Similarly, if a Conv2d layer has 96 filters (C_out=96_ and ) MAC array height = 64, $utilization = 96/(2*64) = 75\%$

If both the width and channel dimensions are not aligned to MAC array geometry, then the utilization factors will multiply. For e.g. if 

- W_out = 224
- C_out = 96
- MAC Array is 64x64

$Utilization = 87.5\% * 75\% = 65.625\%$ 



Low utilization results in lower performance, while we are still incurring the hardware cost and power consumption. Therefore, a good NPU shall achieve not only high TOPS spec, but also high utilization of those TOPS.  

If the MAC array size is increased to 256x256, the utilization for this layer decreases to $(224/256) x (96/256) = 32.81\%$. This shows that bigger MAC arrays result in high TOPS spec, but may have poor utilization if workload is not matched to MAC array size.



> Do note that MAC array utilization does not depend on C_in. This is important because in most CNNs, the first layer has only 1 input channel (grayscale image) or 3 input channels (RGB/YUV image). If MAC utilization depended on C_in, then the first layer would have had very low utilization for large MAC arrays needed by high performance NPUs. This is one of the strengths of the OPA based Matrix Accelerator.    





### **What happens if C_in or C_out is too big?**

In this case, we use NCHWc layout. ***More details : TBD***

 

### **Other types of 2D convolutions**

So far we have considered the simplest form of Conv2d. There are some variations which can potentially have a performance impact. Let’s look at some of them

 

#### **Pointwise Convolutions**

The only difference is that the kernel size, k = 1. Has no impact on performance. The middle loops will run just once. 

         for i in range(k):
          for j in range(k):

We are not exploiting any parallelism in this dimension, and therefore there is no loss in performance.



#### **Strided Convolutions**

In regular convolution, kernel slides one element at a time. Strided convolutions have additional parameters $s_x$ and $s_y$ which make the kernel slide by a larger offset. It requires  $s_x$ * $s_y$  fewer computations and results in the output image being downsampled by  $s_x$ * $s_y$ . The challenge is the required input elements are not contiguous in memory. We need to skip $s_x-1$ elements after reading every element in input tensor and we can not use unit-stride load instructions. 

This is a load problem more than a compute problem. RISC-V matrix extension supports instruction *mlsae* for strided load of A matrix. This instruction selectively pick the required operands from memory to registers. Once the operands are in registers, convolution operation will be done as normal. The performance impact comes from load operation. Strided loads are slower than unit-stride loads by a factor of s. If load bandwidth is on the critical path, it may stall the MPU and cause performance degradation.



#### **Dilated Convolutions**

In dilated convolution, the kernel has "holes" which increases it's receptive field. For e.g. a 3x3 filter with dilation=2, acts as a 5x5 filter with 16 zeros. One way to implement is to run it as a 5x5 convolution. Recall that the number of ops a Conv2d layer requires is *2\* k\*k \*Ci \* Co \* Wo\*Ho* 

Therefore, a 5x5 convolution takes 25/9 = 2.77 times more ops than a 3x3 convolution. It is faster to execute it as a 3x3 convolution, but skip the input elements which are going to be multiplied with zeros in the original 5x5 filter.

This can also be solved by strided load. There is a performance impact, but overall it will be faster than running it as a 5x5 convolution.  



#### **Deformable Convolutions**

Dilated convolutions expand the receptive field by adding ***fixed*** offsets between input elements. In deformable convolutions, the offsets are not fixed but ***generates*** them from input features using learnable parameters. This allows the convolutional filter to adapt its shape and size dynamically, focusing more effectively on relevant features. 

<img src="https://viso.ai/wp-content/uploads/2024/04/simple-convolution-vs-deformable.png" alt="image of simple-convolution-vs-deformable" style="zoom:50%;" />

To perform a deformable convolution, we must fetch all the input elements before we convolve with the filters. In dilated convolution we were able to do by using strided loads. Deformable convolutions are trickier than that. 

1. First, we a run small CNN to generate offsets for every output element. Offsets can be seen as displacement vectors $\Delta x$ and $\Delta y$, for every x, y output location. 
2. In step1, we get a grid of offsets for every output element. This is essentially a geometric transformation with the grid acting as a back-mapping lookup table.  $\Delta x$ and $\Delta y$, maybe fractional which means we need to interpolate between elements at integer locations. For those familiar with Computer Vision and Image Signal Processing, this is similar to an LDC operation. This operation does not map to matrix accelerator. VPU is the most suitable accelerator to do it via table lookup (permute instructions). For every output, 4 inputs need to be accessed, followed by a bilinear interpolation. 
3. After step2, we get the "deformed" representation of input elements. We can now run the regular convolutional kernel on this representation. 



Deformable convolutions have fewer MACs, but step2 is a costly operation in terms of computations as well as memory access. Also, step2 does not execute on MPU, so it tends to get slower and may become the bottleneck. Therefore, it must be chosen wisely.  



#### **Grouped Convolutions**

Grouped convolutions were originally introduced in AlexNet, with the following benefits

1. Filter groups allowed more efficient model-parallelization across the GPUs
2.  Grouped convolutions learn better representations

Regular convolution convolves each of Co filters across *all the input channels* and concatenates the results to generate Co output channels. In a grouped convolution, input channels and filters are divided into *g* groups. Each group of input channels (of size Ci/g) is convolved with Co/g filters. 



![Convolutional Layer with Filter Groups](https://blog.yani.ai/assets/images/posts/2017-08-10-filter-group-tutorial/filtergroups2.svg)

​						*Source : https://blog.yani.ai/assets/images/posts/2017-08-10-filter-group-tutorial/filtergroups2.svg*



Interestingly, it also results in reduced number of parameters and MAC operations. 

|              | **Standard  Convolution** | **Grouped  Convolution** |
| ------------ | ------------------------- | ------------------------ |
| # Parameters | $Co*Ci*k^2$               | $(Co/g)*Ci*k^2$          |
| #MACs        | $Co*Ci*k^2*Wo*Ho$         | $(Co/g)*(Ci/g)*k^2)*g$   |

 

Each group has $g^2$ fewer MACs than the original convolution, and there are *g* such groups. Hence, the overall MAC count goes down by a factor of *g*.

 

**Does this mean inference time of a grouped convolution should be *g* times lower? Not necessarily.** Recall that while accelerating conv2d on MPU, we used Co as one of the parallelization dimensions. This works well if Co is a multiple of matrix MAC array dimension. In grouped convolution, MAC array utilization will be low if (Co/g) does not comply with this. Consider the following scenario:

 

·     Co = 128

·     Matrix MAC Array = 64x64

·     Number of groups, g = 4

In regular convolution, weight tensor gets divided into two tiles of 64 channels each which fully utilizes the X dimension of MAC array.

 

In grouped convolution, each group has 128/4 = 32 channels. MAC array will be only 50% utilized. Therefore, total number of MACs reduces by a factor of 4, but because of 50% efficiency drop, we expect only a 2x increase in inference speed.

 

For larger MAC arrays the utilization will be even lower. For e.g. 128x128 MAC will achieve 100% utilization on regular conv2d but only 25% on grouped convolution in the scenario above. **This is another example where high “TOPS” spec does not convert to high throughput in real world usecases.**



#### **Depthwise Convolutions**

 Depthwise convolution treats each input channel separately. It applies a different filter to each channel. It can be seen a special case of grouped convolution where number of groups, $g = C_i = C_o$. 

Every group has only 1 filter. MAC array utilization goes too low.

This is one of the rare scenarios where im2col maybe a good idea.



#### **Depthwise Separable Convolutions**

 Logically a two-step process – depthwise convolution, followed by a pointwise convolution.

 

![Comparison of a normal convolution and a depthwise separable convolution. a) Standard convolution with a 3x3 kernel and 3 input channels. The projection of one value is shown from the 3x3x3 (dark blue) input values to 6 colorful outputs which would be 6 output channels. b) Depthwise separable convolution with a 3x3 kernel and 3 input channels. First a depthwise convolution projects 3x3 pixels of each input channel to one corresponding output pixel (matching colors). Then a pointwise convolution uses these 3 output pixels to determine the 6 final output pixels.](https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/depthwise-separable-convolution.png)

​					Comparison of a normal convolution and a depthwise separable convolution

Image Source: https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/depthwise-separable-convolution.png





**Can we do something smarter than two convolutions back to back? How can they be merged? **





#### **Point-Wise Convolution with Residual input layer**

 

What can be done about it: TBD? . < https://software-dl.ti.com/mctools/nnc/mcu/users_guide/layer_configs.html >



#### **Transposed Convolutions**

 TBD







## **Performance Analysis** (Outer Product)

MPU Matrix Array = TxT MAC units

Vector LEN = VL (bits)

 SEW is element width (bits)

2 * padding - dilation * (kernel - 1)

Input : Hi x Wi x Ci

Weights : k x k x Ci x Co

Output: Ho x Wo x Co

$W_o = (W_i + 2**pad - dilation*(k-1) - 1)/stride + 1$

$H_o = (H_i + 2**pad - dilation*(k-1) - 1)/stride + 1$$



To compute one output, we need to read $k^2*C_i$ elements from input and same number of elements from weights. There are Ho x Wo x Co elements to be computed.



|      | Operation          | #Params                                          | #OPs                                       | MPU cycles                                 | VPU Cycles                      | Scalar Cycles | Memory R/W (Bytes)                             | Comments                                                     |
| ---- | :----------------- | ------------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ------------------------------- | ------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| 1    | Conv2d             | $k^2*C_i*C_o$                                    | $2*k^2*W_o*H_o*C_i*C_o$                    | $numops*(SEW)^2/(2*T)$                     | 0                               | 0             | $numops*(SEW)$                                 | Assuming input and weights have same width                   |
| 2    | Point Wise Conv2D  | Same as Conv2d with k=1                          | Same as Conv2d with k=1                    | Same as Conv2d with k=1                    | 0                               | 0             | Same as Conv2d                                 |                                                              |
| 3    | Strided Conv2D     | Same as Conv2d                                   | Same as Conv2d                             | Same as Conv2d                             | 0                               | 0             | Same as Conv2d                                 | MPU cycles based on the assumption that strided load is faster than compute. |
| 4    | Dilated Conv2d     | Same as Conv2d                                   | Same as Conv2d                             | Same as Conv2d                             | 0                               | 0             | Same as Conv2d                                 | MPU cycles based on the assumption that strided load is faster than compute. |
| 5    | Deformable  Conv2d | Conv2d Params + Offset Prediction Network Params | Conv2d Ops + Offset Prediction Network Ops | Conv2d Ops + Offset Prediction Network Ops | Grid sampling and interpolation | 0             | Conv2d Ops + Offset Prediction + Grid Sampling |                                                              |
| 6    | Grouped Conv2d     | $(Co/g)*Ci*k^2$                                  | $2*(Co/g)*(Ci/g)*k^2)*g$                   | $numops*(SEW)^2/(2*T)$                     | 0                               | 0             |                                                |                                                              |
| 7    | Depthwise Conv2d   | $k^2*C_o$                                        | $2*k^2*W_o*H_o*C_o$                        | $numops*(SEW)^2/(2*T)$                     | 0                               | 0             |                                                |                                                              |
| 8    | DWS Conv2d         |                                                  |                                            |                                            |                                 |               |                                                |                                                              |

If the $W_o$ and $C_o$ are multiples of T, then the convolutions can be broken down into integer number of TxT dot products resulting in 100% utilization of MAC arrays. If not, then we need to have some tile dot products with MAC arrays running partially unutilized. 

For grouped convolution, we need to consider $C_o/g$



Mathematically MAC array utilization can be computed as $((M*T)/ceiling((M+T-1)/T)) * ((N*T)/ceiling((N+T-1)/T))$

For large values of T (big MAC arrays), it is not always possible to make M and N aligned to T and therefore MAC array runs at lower utilization. 



> [!NOTE]
>
> The cycle count here assumes that the tiles are in L1 memory whenever MPU needs them. In a typical SoC, the big matrices reside in DDR and tiles must be moved fast enough to L1 memory. This has additional performance challenges which we will discuss in chapter 8.
