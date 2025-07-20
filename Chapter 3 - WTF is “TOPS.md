# Chapter 3 - WTF is “TOPS” and “Flops”? Does it matter?**

[TOC]



## **What is it**

- **FLOPS** = Floating Point Operations Per Second. This is the peak throughput of an NPU/GPU when all the operations are done on floating point operands.  
- **TOPS** = Trillion Operations Per Second. Same as Flops but more generic since the operands maybe floating-point or fixed-point. 

Both of these are metrics used by HW vendors to advertise the computational capability their AI accelerator. However, FLOPS is more relevant for training accelerators while TOPS is used commonly for inference accelerators and almost always refers to INT8 TOPS. The reason being INT8 multipliers are the smallest (compared to INT16 and floating point) and therefore a higher spec can be achieved with a smaller die.



## **How is it calculated**

Number of INT8 operations possible in a MPU + VPU based NPU

 
$$
MPU : 2 * M^2 * F
$$

$$
VPU :2 * (VLEN/8) * F
$$

Where

- M = MPU MAC array size in INT8 elements
- VLEN = VPU Vector Length in bits
- F = Frequency

The multiplier “2” comes from the assumption that a MAC operation can be done in a single cycle. MAC includes a multiply and add, and therefore is equivalent to 2 math operations.

An example of an NPU with 64x64 MAC array and 512-bit VPU operating @ 2GHz has a TOPS rating of
$$
(  (2 * 64^2 ) + (2 * 512/8)) * 2*10^9) = 16.64 TOPS
$$
As you can see, most of the TOPS come from MPU and a small fraction from VPU. Technically, scalar can also run some math operations but the overall contribution is negligible and therefore ignored from TOPS calculations.

 

## **Is it relevant**

The short answer is Yes. This is a relevant metric because it gives an indication of how much computational workload the NPU is capable of if *everything else is perfect*.  What does “everything else” mean? Since the TOPS spec is dominated by MPU throughput achieving high utilization is possible only if

1. MPU always has sufficient parallel workload to utilize M2 multipliers

2. MPU is never starved for data

3. MPU, VPU, Scalar, LSUs and DMAs are perfectly parallelized and MPU is the long pole. [Amdahl's law](https://web.engr.oregonstate.edu/~mjb/cs575/Handouts/speedups.and.amdahls.law.1pp.pdf)  always rules.

   

## **Why are things not perfect?**

Let’s take an example. Shown below is the architecture of Resnet50 which is a Convolutional Neural Net designed for image classification. 

![](C:\work\edgeai_book\resnet50.webp)

​										Figure 7 : Resnet50 Model Architecture

​								*Image Source : https://medium.com/data-science/the-annotated-resnet-50-a6c536034758*



The model takes a RGB image of size 224x224 pixels and outputs class probabilities. If we add up all the math operations in the model, it comes up to roughly 4 Billion MACs = 8 Million Operations per inference. So if we run this network on an 8 TOPS accelerator, we should get 1000 inferences/second, right? 

**Wrong!!!!!!!!!!!!**

Let’s look at a few examples

1. NVidia H200 has 3958 INT8 TOPS. Resnet50 expected performance is 494750 fps. Actual performance (https://developer.nvidia.com/deep-learning-performance-training-inference/ai-inference) is 21253 fps, a utilization of almost *4%*. And this is with batchsize=8. Throughput with batchsize=1 is expected to be much lower.
2.  ([Texas Instruments’ appnote](https://www.ti.com/lit/an/spracz2/spracz2.pdf)) shows that Resnet50 performance on TDA4VM 8TOPS NPU is 162fps. This is *16.2%* of what we would expect.
3. Qualcomm Snapdragon 8 Elite Mobile features a 45 TOPS NPU. Resnet50 expected performance is 5625 fps. Actual performance as per https://aihub.qualcomm.com/models/resnet50 is 1876 fps, which shows about *33%* utilization.



Why is the utilization not 100%? Here are a few reasons

 

1. A real world AI workload will include many operations, not only matrix multiplication. Non-matmul operations (like Add, Pool etc.) must be done on VPU. Pipelined architectures allow VPU and MPU to run concurrently (as in hyper-scalar, VLIW etc.) in which case the long pole becomes the bottleneck. Therefore tuning VPU and MPU design parameters to the workload becomes critical to achieving high TOPS utilization.

 

2. Larger array helps achieve high TOPS rating (remember 2xM2xF) but if the input matrices are small, a big MAC array is not fully utilized and the real work done is a small fraction of the theoretical limit. For e.g. in the model architecture shown above, last few layers have a very low spatial resolution (28x28, 14x14, 7x7). On a 64x64 array, the utilization will be less than 50%, 25% and 12%. Imagine what happens if the marketing manager decides to increase MAC array size to 128x128!!! It will increase TOPS spec by 4x, but the % utilization will drop further, resulting in almost no increase in real world performance.

 

3. And finally the memory wall problem. See the section 11.2 on Arithmetic Intensity. 

 

Processing huge tensors with millions/billions of parameters requires moving the inputs, parameters, intermediate results and final outputs between NPU and memory. As we saw in memory hierarchy section, DDR is the largest but also the slowest. Unfortunately, on-chip memories are not large enough to hold the entire model and input/output tensors. Therefore, model parameters and input/output must be broken down into small “tiles” and moved back and forth between on-chip memory and external DDR. Just to give an example, Resnet-50 example above generates <TBD> Mega Bytes of intermediate data per inference. Running this @30 inferences/second will result in <TBD> GB/s traffic. Sounds scary, right? Now imagine a self-driving car where the input resolution goes to a few Mega Pixels and there are multiple cameras feeding to the NPU cluster simultaneously.

 

Memory bandwidth, both within the chip and between the chip and DDR, is limited. Therefore, data movement sometimes takes longer than compute and NPU has to sit idle waiting for data to arrive.

 

A detailed analysis of hardware design choices and the workload characteristics is recommended for a balanced NPU. Design decisions without considering system-level challenges often result in additional area & power with no improvement in performance. As an example, simply doubling the frequency will double the TOPS spec (remember 2xM2xF) but if nothing is done about memory bandwidth, frequency increase will not result in proportionate increase in real world performance.

 

 In a nutshell, TOPS (or TFlops) is a relevant metric, but it does not paint the complete picture. NPU architecture and the SoC architecture play a big role in system level performance. A more relevant metric is “inferences/second”. ***Inference\*** can refer to a camera image in case of image classification, or a token in case of a language model.

 

Having said that, inferences/sec is not a standardized unit of work since every NN model will have a different computational load, different arithmetic intensity and a different ratio of matmul vs other math ops. Therefore, architecting or selecting the ideal NPU is often very application specific. An NPU which is efficient for a Copilot PC may be extremely inefficient for a self-driving car, even though the TOPS rating maybe the same.

 