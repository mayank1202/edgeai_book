# Chapter 7.6 - Attention Mechanisms on NPU

[TOC]



## **Self-Attention**

Self-Attention, also known as Scaled Dot Product Attention, defined below is the cornerstone of Transformer networks

​				$$ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$ 



In chapter 7.1, we briefly touched upon how ${Q.K^T}$ is done efficiently using transposed dot product. In this chapter we will look at more details and additional challenges.  Finally we will look at different types of attention and wrap up with a performance analysis.



### **Vanilla Attention on Matrix Processing Unit (MPU)**

```
def vanilla_attention_npu(Q,K,V, tile_size):

    N, d = Q.shape
    T = tile_size

    A[N,d] = 0   #Attention score
    S[N,N] = 0   #Intermediate result

    for i in range(0,N,T):
        for j in range(0,N,T):
            for k in range(0,d,T):
                q_tile = Q[i:i+T,k:k+T]
                k_tile = K[j:j+T,k:k+T]
                s_tile = tiled_transposed_outer_product(q_tile,k_tile)
                S[i:i+T,j:j+T] += s_tile

	for i in range(N):
    	S[i] = rowwise_softmax(S[i])

    for i in range(0,N,T):
        for j in range(0,d,T):
            for k in range(0,N,T):
                s_tile = S[i:i+T, k:k+T]
                v_tile = V[k:k+T, j:j+T]
                A[i:i+T,j:j+T] += tiled_outer_product(s_tile, v_tile)


    return A
      
```



The processing sequence is

1. **Step 1** - Compute ${Q.K^T}$ by breaking down into TxT tiles and calling the function *tiled_transposed_outer_produc*t on matrix. The result is stored in matrix S, which is of size NxN.
2. **Step 2** - Normalize the rows of matrix $S$ using *softmax* function. As discussed in chapter 7.4, *softmax* runs on VPU.
3. **Step 3** - Compute $S.V$ by breaking down into TxT tiles and calling the function *transposed_outer_product* on matrix.



This is fairly efficient from a compute point of view if the matrices are large and multiple of tile size, T. However, the challenge is that intermediate output $S$ requires $N^2$ memory. This can grow very fast as N (sequence length) increases. For e.g. N=2048 requires **16MB** on-chip SRAM for 32-bit data. 



Two ways of handling this are

1.  Allocate big enough buffer in on-chip SRAM to hold $S$. This maybe cost prohibitive, and may limit the system to models under a certain sequence length.
2. Or, allocate $S$ in external DDR memory and keep moving *s_tile* back and forth between on-chip and off-chip memory. This requires very high DDR bandwidth, in addition to high bandwidth required by AI inferencing anyway. This can be a major performance bottleneck. 



 [Flash attention](https://arxiv.org/pdf/2205.14135) has been proposed a solution for this problem. Flash Attention reduces memory requirement from $O(N^2)$ to $O(N)$. 

However, Flash attention has the following disadvantages

1. Introduces extra computations for block by block re-normalization. 
2. Is a solution designed for GPUs and does not extend well to matrix based NPUs.



Here we propose an alternate solution.



### **Matrix Friendly Flash Attention on MPU**

```

def mpu_flash_attention(Q,K,V, tile_size):

    N, d = Q.shape
    T = tile_size

    A[N,d] = 0   #Attention score output
    S[T,N] = 0   #Intermediate result. We are allocating memory for only TxN elements

    for i in range(0,N,T):
        S = S*0
        for j in range(0,d,T):
            for k in range(0,N,T):
                q_tile = Q[i:i+T,j:j+T]
                k_tile = K[k:k+T,j:j+T]
                s_tile = tiled_transposed_outer_product(q_tile,k_tile)
                S[0:0+T,k:k+T] += s_tile

        for i in range(T):
            S[i] = rowwise_softmax(S[i])

        for j in range(0,d,T):
            for k in range(0,N,T):
                s_tile = S[0:0+T,k:k+T]
                v_tile = V[k:k+T, j:j+T]
                A[i:i+T,j:j+T] += tiled_outer_product(s_tile, v_tile)

    return A
```



**Changes from vanilla attention**

1. Re-ordered the loop. Combined outer loop for all three steps
2. $S$ computation happens for TxN elements, instead of NxN elements.
   - softmax is done rowwise, and therefore there is no need to compute all N rows before doing softmax.
3. Once $S$ is ready for T rows, we can start $S.V$ computation.



**Advantages of this method:**

1. 100% efficiency, as long as N and d are multiple of T. No partial normalizations and no extra computations for re-normalization.
2. Memory requirement is TxN, which is $O(N)$ since T is a hardware constant and does not increase with model complexity. N=2048, T=64 requires only 512KB SRAM.
3. Memory operations are reduced to bare minimum.
4. Friendly with RISC-V NPU architecture
   - Dot products are done on MPU
   - Softmax done on VPU
   - The two operations can be pipelined so that the longer pole determines the throughput. Usually, softmax can be completely hidden behind dot products.





## **Cross-Attention**

Very similar to Self-Attention, with a slight twist

- self-attention computes the interactions between the different elements of an input sequence.
- cross-attention computes the interactions between the elements of 2 different input sequences.



From a computation point of view, the only difference is that we can no longer assume $d_q = d_v = d_k$.  Well, $d_q$ is still = $d_v$, but $d_k$ can be different.

It does not change the execution flow and does not introduce any other complexity. Self-Attention implementation can be reused. 







## **Performance Analysis**

TBD

