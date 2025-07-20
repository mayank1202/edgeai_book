# Chapter 5 - **Data Types and impact on performance****

[TOC]



## **Overview**

Let’s start by looking at NVidia A100’s performance spec for different data types

<img src="C:\work\edgeai_book\NVIDIA-A100-Spec-1024x473.png" style="zoom:75%;" />

​												NVidia A-100 Spec

​								Source: https://frankdenneman.nl/2022/07/26/training-vs-inference-numerical-precision/



Why does throughput vary with data type?

1. 1. As we observed in our RISC-V Vector and Matrix discussion in chapter 1

   

   
   $$
   VPU : ops/cycle = VLEN*LMUL/SEW
   $$
   

   
   $$
   Matrix : ops/cycle = MLEN*AMUL/(SEW^2)
   $$
   

2. Hardware multiplier size is proportional to square of the element width. Therefore, a 16-bit multiplier is 4x the size of an 8-bit multiplier. A 32-bit multiplier is 4x the size of a 16-bit multiplier i.e. 16x the size of an 8-bit multiplier. And a 64-bit multiplier, well you guessed it right, is 64x the size of an 8-bit multiplier.

   

3. Floating point math units are big and power hungry. [IEEE 754 standard](http://www.dsc.ufcg.edu.br/~cnum/modulos/Modulo2/IEEE754_2008.pdf) defines many corner-case scenarios, non-ordinary values, rounding modes and exceptions which the FPU hardware must handle. As per the paper https://arxiv.org/pdf/2303.17951, FP8 hardware is 53% higher gate count than INT8.



The exact relationship between ops/cycle is dependent on the hardware architecture, but is almost always inversely proportionate to the element width. This means that lower precision types result in higher throughput. But there is no free lunch. Lower precision usually comes at the cost of reduced accuracy and there is always a balance to be struck between throughput vs accuracy. We will discuss this in more detail in the chapter on quantization.



The table below lists some of the common formats in AI inferencing. This is by no means a comprehensive list. 

## **Integer formats**

  

| **Format** | **Description**           | **Comments**                                                 |
| ---------- | ------------------------- | ------------------------------------------------------------ |
| (U)INT32   | (unsigned) 32-bit integer | Integer formats, can also be used to represent factional fixed point numbers by using a scaling factor. |
| (U)INT16   | (unsigned) 16-bit integer |                                                              |
| (U)INT8    | (unsigned) 8-bit integer  |                                                              |
| (U)INT4    | (unsigned) 4-bit integer  | Not a standard format, but gaining popularity. Reduces weight memory when two weights are packed in one byte. |

 

## **Floating point formats**

 

| **Format**                                    | **Description**                                              | **Comments**                                                 |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| IEEE FP64                                     | Double Precision FP                                          | Not commonly used in inferencing, or even training.          |
| IEEE FP32                                     | Single Precision FP                                          | Most common format in training, though new formats are gaining traction. |
| IEEE FP16                                     | Half Precision FP                                            | Faster than FP32, but BF16 is giving it a run for the money  |
| Brain Float16                                 | 1 Sign + 5 Exp + 7 Mantissa bits. Same dynamic range as FP32, but lower precision. **Considered a drop-in replacement for FP32.** | Not a standard format, but gaining popularity. **Half the size of FP32 with zero accuracy loss** since AI inference is more sensitive to dynamic range and less to precision. |
| Tensor Float 32                               | 1 Sign + 8 Exp + 10 Mantissa bits                            | 19-bit data within a 32-bit word.  Enables AI training to use tensor cores on NVidia GPUs. |
| FP8-E4M3                                      | 8-bit Floating Point  1 Sign + 4 Exp + 3 Mantissa bits       | Balanced range and precision  Not a standard format, but gaining popularity. |
| FP8-E5M2                                      | 8-bit Floating Point  1 Sign + 5 Exp + 2 Mantissa bits       | Wider range, lower precision  Not a standard format, but gaining popularity. |
| Block Float  https://arxiv.org/pdf/1709.07776 | Hybrid of floating-point and fixed-point arithmetic where a block  of “n” numbers shares an exponent.      FP representation of *1+Lm +Le* bits is compressed to *1 + Lm + Le/n* | **Allows all multiply-accumulate operations to be carried out in fixed-point.** |
| FP4-E2M1                                      | 1 Sign + 2 Exp + 1 Mantissa bit                              | Ultra-low precision. Optimized for weight storage in LLMs and ViTs. |

 

# 

