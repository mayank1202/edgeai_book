# Chapter 7.3 - **How are activation functions accelerated on NPU**

[TOC]



## ReLu (Rectified Linear Unit)

Let's start with the simplest of them all.

$**f(x) = max(0,x)**$

This is essentially a saturation operation.



ReLu is usually preceded with a GeMM or a Convolution operation. As we have seen in chapters 7.1 and 7,2  GeMM and Convolutions are broken down into tiled_dot_products which are further broken down into multiply-accumulate instructions. Usually, dot product accumulation results in widening of data type. Before writing to memory, the accumulated results must be narrowed back into original precision. RISC-V matrix supports reduction instructions which can also do rounding and saturation. 



Thus, ReLu can be fused with the preceding layer and can be done free of cost.



## Other Activation Functions

| Sigmoid             | $f(x) = \frac{1} {1 + e^{-x}}$                               | <img src="https://framerusercontent.com/images/xgEeJAqdwVqAc4pSJBwf9FuGUM.jpg" alt="Sigmoid Function" style="zoom:25%;" /> |
| :------------------ | ------------------------------------------------------------ | :----------------------------------------------------------- |
| **tanh**            | $f(x) =  ({e^x - e^{-x}})/ ({e^x + e^{-x}})$                 | <img src="https://framerusercontent.com/images/bLhT3B3Jbm2NHT5t3X38popWZLw.jpg" alt="Tanh Function (Hyperbolic Tangent)" style="zoom:25%;" /> |
| **Leaky Relu**      | **$f(x) = max(0.1*x,x)$**                                    | <img src="https://framerusercontent.com/images/Y269IbBCkCWk5xaZfYAYiY11c8.jpg" alt="Leaky ReLU" style="zoom:25%;" /> |
| **Parametric Relu** | **$f(x) = max(a*x,x)$**                                      | <img src="https://framerusercontent.com/images/iZ6n9VITIhPoDjge1t4mnT83dDc.jpg" alt="Parametric ReLU" style="zoom:25%;" /> |
| **ELU**             | **$f(x) = x$**  for x>=0 ; **$f(x) = \alpha *(e^x-1)$** for x<0 | <img src="https://framerusercontent.com/images/H5ytjeTKvyjahj7YdOW9UXcPDY4.jpg" alt="ELU Activation Function" style="zoom:25%;" /> |
| **GeLu**            | **$f(x) = 0.5*x*(1+tanh(sqrt(2/\pi*(x+0.044715*x^3))))$**    | <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/12/Plot_of_GELU.png" alt="GELU function" style="zoom:25%;" /> |



All the activation functions , take an input x and return a value y. softmax is different since it operates on a set of values and returns normalized version of x. softmax deserves it's own special chapter and we will revisit it later.



For all other activation functions, we have three options  

1. Compute them on CPU (scalar) using math libraries.
2. Compute them offline and store in a lookup table.
3. Design a hardware block for each of the activation function.



Option#3 will be the fastest, but is impractical since there are way too many functions and new ones may emerge in future.

Option#1 is the most future-proof, but is also the slowest.

Option#2 seems to be the most well balanced approach, but it has it's pros and cons:

**Pros of Lookup Table approach**

1. Future Proof - Any activation function can be computed offline and stored in memory
2. Vector has permute instructions which can lookup VL elements in parallel.



**Cons of Lookup Table approach** 

1. Vector permute instructions are slower than math instructions. However, it is still expected to be faster than scalar. 
2. For higher precision modes, lookup table can be quite big and may not fit in SRAM/TCM. Workaround for this is to keep the tables small and interpolate between adjacent entries. As can be seen in the table above, all the activation functions have a linear shape in a small range, because of which a simple linear interpolation would give good results.



## Best Practices for Edge Inference

To get best performance, it is a good practice to modify NN graph to use ReLu as much as possible during inferencing. Other activations are very useful in training, but at inference time their benefits are limited and most of them can be replaced with ReLu with minimal loss to accuracy. 



## 
