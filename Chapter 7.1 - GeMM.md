# Chapter 7.1 - **How are GeMM operations accelerated on NPU**

[TOC]



## **General matrix multiply (GeMM)**

GeMM is the foundation for many operators in an AI graph. Some examples are Fully Connected aka dense layers, Feed Forward Networks and Multi-Layer Perceptron. A GeMM operator has a form
$$
C = A.B
$$
Where, A, B and C are matrices of shape MxK, KxN and MxN respectively. 

Let’s see how this operation will be executed on hardware. We will look at three scenarios – 

- The first one is a simple scalar CPU with no vector or matrix, 
- Second is Scalar + VPU and 
- Third will be a full blown NPU with Scalar + Vector + MPU. 

In all scenarios, we will assume that A, B, C are large matrices and reside in external memory DDR. Therefore, compute performance will be sensitive to memory b/w and access patterns.

### **GeMM on a scalar**

####  **Inner Product Method**

A GeMM operation of form C = A.B can be implemented as 3-level nested for loop, where output element {i,j} is a dot product of ith row of A and jth row of B.

`for` m=0:M` 
    for n=0:N                           
        for k=0:K                       
            C[m,n] += A[m,k] * B[k,n]`

The inner loop does the following

1. Loads A[m,k] and B[k,j] from memory to CPU register files.
2. Computes the product and accumulates into another register. Most CPUs have FMA (fused-multiply-add) instruction which can do multiply & add in a single step.

 

After the inner loop is complete, C[m,n] is ready to be written out to memory. CPU may do a reduction or saturation to convert it to a lower data type. The inner loop is repeated for all *MxN* outputs in matrix *C*. Hence, the full loop takes *M\*N*K* FMAs. This method is known as **Inner Product** method. It generates a single full sum at a time, with no merging of partial sums. This requires a hardware intersection unit to align effectual inputs. This is fairly efficient from computation point of view, ***if\*** all load/store and other instructions can be pipelined then the runtime will be dominated by FMA cycles.

But that’s a big **if**. The biggest problem we see here is access pattern of matrix B resulting is an extremely high load time of B[k,n]. To understand this better, let’s note that the change variable in the inner loop is *k*, which means that for successive values of *k**,*** B[k,n] is read from a different row in the matrix. <u>*This results in highly inefficient memory access*</u> for two reasons

1. B[k,n] and B[k+1,n] are not in the same cache line. Hence caching or prefetching results in worse performance than uncached access.

 

2. The matrix is stored in DDR is row-major layout. B[k,n] and B[k+1,n] reside in different DDR pages. This results in extremely inefficient DDR access and low utilization of DDR bandwidth. For more details on DDR bandwidth utilization, please see Appendix 11.1.

 

**Is there a better way? There are actually two.**



#### **Gustavson’s method**

We can move inner loop to middle as shown below

`for` m=0:M` 
    for k=0:K                           
        for n=0:N                       
            C[m,n] += A[m,k] * B[k,n]`



The inner loop generates partial sums of *n* output elements. *n* final outputs will be ready after the end of the middle loop.  This **improves spatial locality in input matrix B**, but side-effect is that we need *N* registers for accumulation.





#### **Outer Product Method**



We can move inner loop one more level outside as shown below

`for` k=0:K` 
    for m=0:m                           
        for n=0:N                       
            C[m,n] += A[m,k] * B[k,n]`

The inner loop does a partial update of mx*n* output elements. mx*n* final outputs will be ready after the end of the outer loop.  This improves spatial locality in matrix B but **also increases data reuse for matrix A**. Note that A[m,k] is independent of the inner loop change variable *n*. Hence, a row of A needs to be loaded only once in the middle loop and gets reused for all iterations of the inner loop. This reduces number of A loads by a factor of k.

But nothing comes free. The cost of doing this reordering is that we need mx*n* registers for accumulation. This is not economical on CPU scalar, but we will revisit this later when we discuss matmul on MPUs.



### **GeMM on a VPU**

We start with Gustavson’s method and modify it for parallelization along the "N" axis.

`for` m=0:M` 
    for k=0:K                           
        for n=0:N/VL                       
            C[m,n:**n+VL**] += A[m,k] * B[k,n:n+VL]`



The inner loop does *VL* FMAs in parallel. Remember from chapter 1 
$$
VL <= VLMAX = (VLEN*LMUL)/SEW
$$
For VLEN=512, SEW=8, VPU can process 64 FMAs (=128 ops) per cycle. However, achieving this throughput requires load/store to run at least as fast as compute. This is not possible if A, B and C matrices reside in DDR. The SoC may have an internal SRAM (or TCM) but if the matrices are too big, SRAM may not be enough. 

**Solution** – Tiling comes to our rescue. We divide matrices into smaller submatrices. In general, GEMM of form **C = A.B** can be broken down into submatrices (aka tiles) of size TxT, such that

- A is divided into [M/T] * [K/T] tiles

- B is divided into [K/T] * [N/T] tiles

  *******************Diagram : TBD***********************

Total number of tile = (M/T)*(N/T)*(K/T)

C can be computed tile-wise as C_Tile[i,j] = Σdotproduct(A_tile[i,k] , B_tile[k,n] )

- dotproduct of A and B tiles is an atomic operation on MPU.
- Σ is done by accumulating through all the tiles from k = 0 -> K/T.

Tile size is chosen such that all the input and output tiles fit in internal SRAM. 



### **GeMM on an MPU**

 We will discuss two ways of doing it - inner product and outer product, and we will discuss pros and cons of each. 

 

**Method1: Inner Product**: Use Inner Product method and extend the parallelization to two dimensions. We also set tile size to match with the matrix panel size on MPU. Let’s assume square tiles of size TxT.

 

`for m=0:M:T` 
  `for n=0:N:T                           
    for k=0:K:T`  

​     `tile_A = A[m:m+T,k:k+T]`   

​     `tile_B = B[k:k+T, n:n+T]`   

​     `C[m:m+T,n:n+T] = tiled_inner_product(tile_A, tile_B)`   

​                    

*tiled_inner_product* is computed row by row, taking the inner product of a row of A, with multiple columns of B. 

  

`f`or x=0:T` 
  C_tile[x] += inner(tile_A[x], tile_B)`   

​                     

*A[x]* is the xth row of tile A. “*inner*” is a **vector-matrix** dot product function which is done on an Inner-Product based MPU hardware in a single cycle. The output C_tile is generated row by row. The entire tile is computed in T cycles. Thus 2T^3 arithmetic operations are done in T cycles, resulting in a speedup of 2T^2.

> **Need to double check : In this method, to compute T outputs, MPU needs to load 1 row of A and T columns of B, each of size T elements. This is repeated T times to compute the tile. Thus, total number of loads per tile = T*(T+T2) = T2 + T3**

Not bad, but can we do better?



**Option 2: Outer Product**: 

Now let’s take the outer product method and extend the parallelization to two dimensions, as follows

 

`for m=0:M:T` 
  `for n=0:N:T                           
    for k=0:K:T`  

​     `tile_A = A[m:m+T,k:k+T]`   

​     `tile_B = B[k:k+T, n:n+T]`   

​     `C[m:m+T,n:n+T] = tiled_outer_product(tile_A, tile_B)`   

​                    

 

*tiled_outer_product* is computed step by step for all the elements in the tile, taking the outer product of **one** **column of tile_A**, **with one row of tile_B**. 

 

`f`or x=0:T` 
  C_tile += outer(tile_A[:,x], tile_B[x,:])`   



 *A[:,x]* is the xth column of tile A. *B[x,:]* is the xth row of tile B. “*outer*” is a vector-vector outer product function which is done on an Outer-Product based MPU hardware in a single cycle. This generates TxT partial outputs which are accumulated for all values of x. The entire tile is computed in T cycles. Thus 2T^3 arithmetic operations are done in T cycles, resulting in a speedup of 2T^2. Same as inner product method.

 

Then what’s the difference?

> Need to double check : The difference lies in the number of loads. Every iteration of the loop requires 2T loads. This is repeated T times for a total of T2 loads. Significant improvement from T2 + T3 in case of IME. 
>
>  	

As always, nothing comes for free. Outer product requires TxT accumulator array, which can be quite costly for large T values. This method has a higher overhead, but it gives system designers to create a high performance NPU with balanced load/store with compute. Therefore it is a good choice for high performance AI accelerators.

For rest of this book we will assume outer product based MPU for performance analysis. 

> TBD : Balanced load/store with compute analysis

 

