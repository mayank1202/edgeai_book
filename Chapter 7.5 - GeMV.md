# Chapter 7.5 - **How are GeMV operations accelerated on NPU**

[TOC]



## **General Matrix - Vector Multiplication (GeMV)**

An operation $y = A.x$ where $A$ is a MxN matrix and $x$ is a Nx1 vector is called GeMV.  The output $y$ is a Mx1 vector.

In theory, GeMV is a special case of GeMM where the second matrix has only 1 row. However, from hardware performance point of view, it changes things quite significantly. 



Recall from GeMM chapter, while computing $C=A.B$ matrix utilization  is dependent on the number of columns in  $A$ and number of rows in $B$. If $B$ is a scalar, then matrix utilization goes down to 1/T, where T is the matrix MAC array size.



GeMV has become a major bottleneck in newer AI models, specifically the ones based on Transformers. In a LLM model, most of the computational load comes from

1. Input projection
2. Multi Head Attention
3. Feed Forward Network
4. Output projection

These all sound like GeMM based operations, but important point to note is that models which process one token at at time have input of size 1xd, where d is the token dimension. 



However, there are models like Deepseek R1 which uses Multi-Token Prediction. Such models can leverage 2D parallelism in GeMM. 



GeMV can utilize data parallelism only in 1 dimension, therefore VPU can run this at higher efficiency.



### **GeMV on a VPU**



```
for i in range (M):
   accum = 0
   for j in range (0,N,VL):
       accum += A[i,j:j+VL]*B[j:j+VL]          #Element wise multiplication e.g. vfmul.vv
   c[i]  = sum(accum)                          #Reduction, e.g. vredsum instruction
```



VL here is VLEN/SEW i.e. the number of elements processed in parallel. 



The inner loop does the following

1. Initializes a set accumulation register to 0. The number of accumulation registers depend on the data type. For e.g. fixed-point formats result in widening and more registers may be needed.
2. Loads VL elements from $i_{th}$ row of matrix A. These are spatially close, resulting in fast loading.
3. Loads VL elements from vector B. These are also spatially close.
4. Performs elementwise multiplication and accumulates in the accumulation registers.   

 At the end of inner loop, the accumulated registers are summed together using a **<u> vector reduction</u>** instruction. The result is ready to be committed to memory. 



## **Performance Analysis**

For every output element:

1. N/VL instructions for element wise multiplication. Expected 1 cycle/instruction.
2. Two load instructions. If the VPU has two LSUs, loads can be pipelined with compute.
   - <!--If VPU has only 1 LSU, there will be a performance impact. Some of it can be minimized by using more vector registers and increasing data reuse.-->  
3. One reduction instruction. May take longer than 1 cycle. Worst case VL cycles.
4. One store instruction. Can be pipelined with compute.



Total cycles for GemV = $M*(VL + N/VL)$

