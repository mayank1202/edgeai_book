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



```
 for m in range (M):                           
    for n in range (N):                 
        for in range (K):
           C[m,n] += A[m,k] * B[k,n]
```



The inner loop does the following

1. Loads A[m,k] and B[k,j] from memory to CPU register files.
2. Computes the product and accumulates into another register. Most CPUs have FMA (fused-multiply-add) instruction which can do multiply & add in a single step.

 

After the inner loop is complete, C[m,n] is ready to be written out to memory. CPU may do a reduction or saturation to convert it to a lower data type. The inner loop is repeated for all *MxN* outputs in matrix *C*. Hence, the full loop takes M*N*K FMAs. This method is known as **Inner Product** method. It generates a single full sum at a time, with no merging of partial sums. This is fairly efficient from computation point of view, ***if\*** all load/store and other instructions can be pipelined then the runtime will be dominated by FMA cycles.

But that’s a big **if**. The biggest problem we see here is access pattern of matrix B resulting is an extremely high load time of B[k,n]. To understand this better, let’s note that the change variable in the inner loop is *k*, which means that for successive values of *k**,*** B[k,n] is read from a different row in the matrix. <u>*This results in highly inefficient memory access*</u> for two reasons

1. B[k,n] and B[k+1,n] are not in the same cache line. Hence caching or prefetching results in worse performance than uncached access.

 

2. The matrix is stored in DDR is row-major layout. B[k,n] and B[k+1,n] reside in different DDR pages. This results in extremely inefficient DDR access and low utilization of DDR bandwidth. For more details on DDR bandwidth utilization, please see Appendix 11.1.

 

**Is there a better way? There are actually two.**



#### **Gustavson’s method**

We can move inner loop to middle as shown below

```
 for m in range (M):                           
    for k in range (K):                 
        for n in range (N):
           C[m,n] += A[m,k] * B[k,n]
```



The inner loop generates partial sums of *n* output elements. *n* final outputs will be ready after the end of the middle loop.  This **improves spatial locality in input matrix B**, but side-effect is that we need *N* registers for accumulation.





#### **Outer Product Method**



We can move inner loop one more level outside as shown below

```
for k in range (K):
    for m in range (M):
       for n in range (N):
          C[m,n] += A[m,k] * B[k,n]
```



The inner loop does a partial update of mx*n* output elements. mx*n* final outputs will be ready after the end of the outer loop.  This improves spatial locality in matrix B but **also increases data reuse for matrix A**. Note that A[m,k] is independent of the inner loop change variable *n*. Hence, a row of A needs to be loaded only once in the middle loop and gets reused for all iterations of the inner loop. This reduces number of A loads by a factor of k.

But nothing comes free. The cost of doing this reordering is that we need mx*n* registers for accumulation. This is not economical on CPU scalar, but we will revisit this later when we discuss matmul on MPUs.



### **GeMM on a VPU**

We start with Gustavson’s method and modify it for parallelization along the "N" axis.

```
for m in range (M):
   for k in range (K):
       for n in range (0,N,VL):
          C[m,n] += A[m,k] * B[k,n:n+VL]
```



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

 Assuming, M, N and K are large, first step is to divide them into smaller "*tiles*" which can fit in MPU registers. For this discussion, we assume square tiles of size TxT, although in practice, tile size and shape depends on MPU implementation. 



This is how big matrices are divided into TxT tiles. 

```
 for m in range (0,M,T):
    for n in range (0,N,T):
        for k in range (K):
           tile_A = A[m:m+T,k:k+T]
           tile_B = B[k:k+T, n:n+T]
           tile_C += tiled_dot_product(tile_A, tile_B)
        C[m:m+T,n:n+T] = tile_C
           
```

*tiled_dot_product* is the micro-kernel running on the MPU, computing the dot product of tile_A, tile_B. 

This micro-kernel is the building block of all matrix based AI operations including fully connected layers, convolutions and more. Therefore, optimizing this micro-kernel is key to getting best performance per watt and per area.

Let's recall a few things before we go into performance analysis

1.  Matrix multiply has computation complexity of $O(N^3)$.
2. VPU offers $O(N)$ parallelism.
3. MPU offers $(ON^2)$ parallelism. Execution time is $O(N)$.
4. Matrix load is an $(ON^2)$ operation, since we have to load matrices of dimension NxN.
5. This means that load takes much longer than compute, unless we do something about it. 



Let's look at one of the implementations and analyze the performance of the micro-kernel

### **Outer Product Performance Analysis**

Reference : https://www.securerisc.org/RISCVIME/index.html

Assumptions

1. Load bandwidth, B = MLEN/SEW bits/cycle
2. Matrix MAC Array Size = TxT, where T= MLEN/SEW

For e.g. MLEN = 512, SEW=8 has 64x64 MAC array and 512-b = 64-bytes/cycle load bandwidth



The micro-kernel tiled_dot_product is implemented as follows

1. Load a row of tile_A. 64-bytes. 1 cycle.
2. Load a column of tile_B. 64-bytes. 1 cycle.
3. Compute outer product and accumulate in tile_C. 1 cycle.

Repeat this T times to complete TxT dot product.



**Pseudocode**

```
def tiled_dot_product(tile_A, tile_B):
    for i in range (T):        #T is a hardware constant  
        mload ms1, tile_A[i,:] #Load i_th row of tile A into register ms1
        mload ms2, tile_B[:,i] #Load i_th col of tile B into register ms2
        outer acc ms1 ms2      #Compute outer product and accumulate in register acc     
```

Total #cycles = 3T. Average throughput = $2T^3/3$ ops/cycle

This is much lower than ($2*T^2$), which is what we expected since we have TxT MAC array. The reason is that load and compute are being done sequentially. Since MPU's LSU and EXU are separate units, we should be able to do at least one load in parallel with compute.

Let's modify the pseudocode accordingly



```
def tiled_dot_product(tile_A, tile_B):
    mload ms1, tile_A[0,:] ; 
    mload ms2, tile_B[:,0] 
	for i in range (T):                                   #T is a hardware constant 
        outer acc ms1 ms2      ; mload ms1, tile_A[i+1,:] #Compute and load in parallel
        mload ms2, tile_B[:,i+1]                          #Only load. There is nothing to compute since we are waiting for data
```

We are able to save 1 cycle by hiding it behind outer product computation. 

Total #cycles = 2T. Average throughput = ($T^2$) ops/cycle. Good, but can we do better. 

Turns out we can, as follows



1. Load two rows from A in 2 cycles.
2. Load two columns from B in 2 cycles.
3.  There are 4 outer products to be computed
   1. A0B0
   2. A0B1
   3. A1B0
   4. A1B1

Now there are 4 loads and 4 compute cycles in the loop. Let's modify the micro-kernel as follows 

```
def tiled_dot_product(tile_A, tile_B):
    mload ms1, tile_A[0,:]
    mload ms2, tile_B[:,0] 
	for i in range (T):                                       #T is a hardware constant 
        outer acc00 ms1 ms2      ; mload ms3, tile_A[2*i+1,:] #Compute and load in parallel
        outer acc01 ms3 ms2      ; mload ms4, tile_B[:,2*i+1] #Compute and load in parallel
        outer acc10 ms1 ms4      ; mload ms1, tile_A[2*i+2,:] #Compute and load in parallel
        outer acc11 ms3 ms4      ; mload ms2, tile_B[:,2*i+2] #Compute and load in parallel

```

 Total #cycles = 4T, but this  micro-kernel computes dot product of 2T x 2T tiles, which is equivalent of 8T3 operations . 

Average throughput =$8T^3/4T = 2T^2$   ops/cycle. 

**This is the max throughput we could have achieved with a TxT MAC array.** 

However, to achieve this throughput we had to use 4 times the number of accumulators. This is the die area cost we have to pay to get high performance.





For a frequency ***F*** this architecture gives us peak performance of $2T^2 * F$ INT8 TOPS. 

For e.g. with MLEN = 512, SEW=8, Freq=2 GHz, we get 

$T = 512/8 = 64$

$Peak Perf = 2 * 64 * 64 * 2GHz = 16 TOPS$ 

 

> For rest of this book we will assume outer product based MPU for performance analysis.

 

> We assumed square TxT tiles here, but we can generalize it to MxK and KxN inputs and MxN outputs. MAC array will be of size MxN, giving a throughput of $2*M*N$ ops/cycle. Do note that this is independent of K. This will be important when we map other operations (like Conv2d) to MPU.

 



