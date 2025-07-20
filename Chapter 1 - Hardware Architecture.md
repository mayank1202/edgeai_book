# Chapter 1 - Hardware Architecture

[TOC]



## Hardware Block Diagram

​                               

 <img src="C:\work\edgeai_book\NPU_BD.jpg" style="zoom:50%;" />



Figure 1 : Block Diagram of a RISC-V based NPU 

<img src="C:\work\edgeai_book\SoC_BD.jpg" style="zoom:50%;" />

Figure 2 : Example of a SoC with an Apps CPU Cluster, NPU cluster, On-Chip SRAM and External Memory Interface to DDR

 

 

In the architecture assumed here, the NPU acts as an accelerator which offloads the Apps CPU from the heavy computational load. An example of an Apps CPU is ARM Cortex-A, or a 64-bit RISC-V core, typically running an Operating System like Linux, QNX or RTOS. In the typical usage model, Apps CPU hosts the application (e.g. Face ID), receives the data (e.g. camera frames) and sends it to the NPU for inference computation (e.g. MTCNN face recognition). Apps CPU then goes back to running OS and apps while the NPU is running math heavy computations. Once the NPU is done, it informs the Apps CPU and the processed results are passed on to the application which requested it.

In the meanwhile, NPU will need additional data from off-chip memory (e.g. DDR) or from on-chip SRAM. It accesses the memory using its own data movement engines so that the Apps CPU is not disturbed. 

The choice of architecture is motivated by the following principles:

In a typical AI model, computational load is heavily dominated by matrix multiplication operations. Matrix Processing Unit (**MPU**) is a hardware block custom designed to process 2D matrices, with **M**x**N** operations in parallel. Hence it can perform these operations at a very high speed and low power. *M and N govern the matrix MAC array dimension and are important design decisions affecting PPA.*

 

- Since MPU is custom designed for matrix multiplication, it cannot process other operations like addition, min/max etc. These operations are offloaded to a Vector (**VPU**) which offers 1-dimensional parallelism.

  

- VPU has a limited instruction set, and many operations are not good candidates for data parallelism. This is where **scalar** comes handy. Additionally, scalar core is responsible to run control software and moving data from cache/TCM to register files.

   

- AI operations are memory intensive. To achieve full utilization of compute elements, data must be moved in/out of registers fast enough. **TCM** is a memory sitting close to the CPU, often with same latency as a cache hit. However, it has lower complexity than cache because of no cache management and coherence to worry about.

   

- Data Movement Engine (DME) is responsible for bringing data from On Chip SRAM to TCM. DME takes care of bringing small tiles from big tensors which may be stored in a multi-dimensional layout.

  

In the next section, we take a closer look at each of the hardware modules.

## Scalar Engine

As described earlier, scalar is responsible for graph execution and issuing instructions for scalar’s compute elements (ALU, FPU etc.) as well as VPU and MPU. In some cases, scalar’s LSU (Load store unit) may also be responsible for moving data from TCM/cache to/from VPU’s register files. Therefore, the scalar plays a pivotal role in AI graph execution. Further, there are many operations which do not fit within VPU or MPU’s instruction set. Hence, these must be executed on the scalar. Some examples of such operations are

- Non Maximal Suppression
- ArgMax, ArgMin
- AffineGrid
- Logical Operators (And/Or/Xor)
- Reshape
- Flatten
- ..and many more

For best performance, it is expected that an AI graph will have very few of such operations. For e.g. NMS is used at the end of the graph to find the best bounding box. Most of the heavy lifting is done by MPU and VPU, therefore the scalar engine does not need be an ultra-high performance CPU core, but there are a few attributes which can help get maximum performance from the NPU

1. Instruction level parallelism – helps in executing load/store, scalar, vector and matrix instructions in a pipelined fashion, keeping all HW blocks occupied as much as possible. 
2. Multi-threading – Low latency context switches.
3. FPU – Floating Point Unit for generating final probabilities and classification scores.

Some of the NPUs use a DSP as the scalar engine, which commonly use VLIW for Instruction Level Parallelism. Similar results can be achieved on RISC-V cores with hyper-scalar pipelines.

* [A comparison of VLIW vs SMT](https://www.cs.umd.edu/~meesh/411/CA-online/chapter/multiple-issue-processors-ii/index.html)* 

Some DSPs also support multi-threading, for e.g. Qualcomm Snapdragon 8 Gen 2, Hexagon has 6-way SMT. 

[RISC-V supports up to 8-way SMT](https://www.youtube.com/watch?v=QR4niawYSVI ) 

 

## Vector Processing Unit (VPU)

Vector processing is a type of data level parallelism where a single instruction operates on multiple data elements simultaneously. In concept, it is very similar to SIMD (e.g. ARM Neon, Intel AVX-512) but there are some subtle but critical differences. Refer to [RISC-V Vector vs ARM Neon SIMD](RISC-V Vector vs ARM Neon SIMD) 

Key design time attributes of a VPU based on RISC-V are

1. **VLEN** – Register width (in bits).
2. **ELEN** – Widest supported element size.
3. **LMUL** – Lane multiplier for register grouping. Allows packing multiple registers together to operate more number of elements in parallel. 

**SEW** (Selected Element Width) is a runtime configuration to choose the bitwidth of the operands. This gives us the maximum number of elements which can be processed together:

**VLMAX = (VLEN\*LMUL)/SEW**

VLEN, ELEN and LMUL are important design decisions, impacting performance, die area and therefore the cost & power of the VPU.

Vector engines offer not only faster execution but also code size reduction. Let’s take an example of adding two arrays of length n.

void vvaddint32(size_t n, const int*x, const int*y, int*z){ 

`for (size_t i=0; i<n; i++)`

`z[i]=x[i]+y[i];` 

​    `}`

Figure 3 : C code of array addition on scalar

  This loop has n additions, increments of i, comparisons between i and n, and branching logic. When compiled, it will result in 36 instructions. 

 

On RISC-V Vector, it compiles into 11 instructions as shown below 

`vvaddint32:`

  `vsetvli t0, a0, e32, ta, ma # Set vector length based on 32-bit vectors`

  `vle32.v v0, (a1)     # Get first vector`

   `sub a0, a0, t0     # Decrement number done`

   `slli t0, t0, 2     # Multiply number done by 4 bytes`

   `add a1, a1, t0     # Bump pointer`

  `vle32.v v1, (a2)     # Get second vector`

   `add a2, a2, t0     # Bump pointer`

  `vadd.vv v2, v0, v1    # Sum vectors`

  `vse32.v v2, (a3)     # Store result`

   `add a3, a3, t0     # Bump pointer`

   `bnez a0, vvaddint32  # Loop back`

   `ret          # Finished`

Figure 4 : Assembly code of array addition on RISC-V Vector

Source : https://github.com/riscvarchive/riscv-v-spec/blob/master/example/vvaddint32.s 

We see more than 3x improvement in code density. On a 256-bit vector, vadd.vv instruction processes 8 elements at a time since data type is int32. 

A lot of AI operations like “*add*”, “*concat*” etc. can be done effectively on vector. RISC-V Vector also supports permute instructions which can be used for table lookup. This is extremely helpful in running ops which include exponentiation and logarithmic functions. Softmax, used heavily in transformer networks, is a good example. 

 



 

## Matrix Processing Unit (MPU)

Also known as “Tensor Processor” on some NPUs, this block offers data parallelism across two dimensions compared to a vector which offers 1D parallelism. Matrix array size is a key design decision. Larger matrices can help boost performance, but result in higher area and power. When matrix array becomes very large, feeding it with valid input can become challenging and may cause under-utilization of processing elements. We will visit this topic in more detail in section 8.1. 

Basic operation of an MPU is explained below, 

 <img src="C:\work\edgeai_book\Matrix_BD.png" style="zoom:50%;" />

Figure 5 : MPU matmul

There are several matrix extensions proposed in RISC-V forum. The two most prominent are

1. **IME (Integrated Matrix Extension)** – Closely coupled with Vector. Matrix and Vector registers are shared. Matrix introduces no additional architecture state, making it easier for adoption in OSes (e.g. Linux). MAC array size is constrained to vector length, and it becomes a challenge to process computationally intensive workloads.  Despite these limitations, this architecture is well suited for AI enabled Apps CPUs.

 

2. **AME (Attached Matrix Extension)** – Decoupled from Vector. Matrix has its own registers and accumulators. Designers have more freedom to implement schemes like “outer-product matmul” on larger MAC arrays. More suitable for NPU accelerators where high performance (/watt and /mm2) is the goal.

A lot of topics we will discuss in this book are applicable to both the architectures, but the authors assume AME wherever there is a divergence.

[RISC-V AME spec 0.5b](https://lists.riscv.org/g/tech-attached-matrix-extension/attachment/210/1) defines:

1. **ELEN** – Widest supported element size. Similar to ELEN in Vector.
2. **MLEN** – Matrix tile size (in bits).
3. **RLEN** – Matrix row size (in bits).
4. **AMUL** - multiple of length for matrix accumulation registers. Similar to LMUL in vector. AMUL can be used to group registers to operate on wider elements, up to 8x. TBD : Confirm

**SEW** = Selected element width, similar to as in Vector.

An Accumulation Register has MLEN*AMUL bits of state, where each row has RLEN*AMUL bits. As a result, there are MLEN/RLEN rows for each accumulation register in logic.

This leads to the following constraints

1. **TMMAX**, max value of m in Figure 5 = MLEN/RLEN
2. **TNMAX**, max value of n in Figure 5 = RLEN/SEW
3. **TKMAX**, max value of k in Figure 5 = min(TMMAX, TNMAX)

As an example, for MLEN=512, RLEN=64, SEW=8

TMMAX = 8, TNMAX = 8, TKMAX = 8. This means that MPU is designed to perform 8x8x8 matrix multiplication on INT8/FP8 data.



Before MPU can process the matrices, they must be loaded to Matrix Register File. This is done by matrix LSU. Since register file space is small and precious, entire matrices are not loaded at once. Instead big matrices are divided into tiles of size *mtile_m x mtile_k x mtile_n*. The loads are done row-wise for matrix A and column-wise for matrix B.

The basic operational steps for compute dot product of 1 tile are

1. Load row #i, from tile A.
2. Load column #j from tile B.
3. Compute outer product. The result in accumulated in matrix registers.
4. Repeat steps 1-3 *mtile_k* number of times.

Step #3 above is where computation happens. A typical MPU can execute this in a single cycle. Hence, the loop above computes tile A and B dot product in *mtile_m* x *mtile_n* cycles. 

The total number of computations for a tile are *mtile_m* * *mtile_k* * *mtile_n* multiplies and adds i.e. total **2\* \*mtile_m\* \* \*mtile_k\* \* \*mtile_n\*** math operations, done in ***mtile_k\*** cycles. Thus MPU achieves a speedup of **2\* \*mtile_m\* \* \*mtile_n\*** ops/cycle. Full throughput is achieved when

·     *mtile_m =* TMMAX/SEW

·     *mtile_k =* TKMAX 

·     *mtile_n =* TNMAX

Substituting the values of TMMAX, TNMAX and TKMAX, we compute maximum throughput achievable = 

2* ((MLEN/RLEN)/SEW) * (RLEN/SEW) = **2\*MLEN/SEW2**

A MPU with 4096-bit MLEN, running at 1GHz, operating on 8-bit data has a peak performance rating of 

*2 \* 642 \* 1G = 8 Trillion Operations Per Second, commonly expressed as **8 TOPS**.*

 

## Memory Hierarchy

 <img src="C:\work\edgeai_book\Mem_hierarchy.png" style="zoom:50%;" />

Figure 6 : Typical Memory Hierarchy in an embedded SoC

A hierarchical memory organization helps in a balanced PPA by arranging different types of memory in layers. The table below shows the speed vs cost vs data movement energy comparison of different types of memories

| **Memory Type**      | **Cost/byte** | **Speed** | **Energy/byte** |
| -------------------- | ------------- | --------- | --------------- |
| Registers            | Highest       | Fastest   | Lowest          |
| TCM/Cache (L1,L2,L3) | High          | Fast      | Low             |
| On Chip RAM/LLC      | Mid           | Mid       | Mid             |
| DDR                  | Low           | Slow      | High            |
| Storage              | Lowest        | Slowest   | Highest         |

 

This places an economical (and also physical) limit on capacities of registers files, caches and on-chip RAM. To achieve best performance, data must be moved in small chunks from big memories to fast memories close to execution units. The data must be evicted when it is no longer needed so that space can be created for the next tile of data. Detailed discussion of different types of memory is outside the scope of this book. However, we will discuss a few important topics which we believe are highly important to understand performance challenges of AI inference.

### TCM vs Cache

A TCM (Tightly Coupled Memory) sits close to the execution units and offers access latency comparable to L1 cache hit. The major differences between TCM and cache are

1. TCMs are mapped in system memory map, while cache is not.
2. Cache can be accessed only by the CPU, while TCMs can be accessed by other masters, including DMA engines in the SoC.
3. TCMs can be multi-port, which allows other cores to access TCM of one core, although at a higher latency.
4. TCMs have lower overheads, since there is no coherence, no eviction policy and no tags to be maintained.

Therefore, TCMs are preferred to caches to store AI data. The execution order in an AI graph is highly predictable and software controlled management is quite effective.

### DDR Bandwidth

DDR bandwidth, expressed in GB/s, is the rate at which data can be transferred between the SoC and the DDR. This is a critical factor in any embedded SoC, but becomes even more critical in AI because of heavy data intensive nature of workloads. If the data cannot be moved fast enough to execution units, it cause stalls which results in lower performance. DDR bandwidth is a shared resource between all the subsystems inside a SoC including Apps CPU, ISP (Image Signal Processor), DSS (Display Sub System), VSS (Video Subsystem), NPU and many more. Therefore an NPU which can minimize off-chip data transfers usually results in higher performance in real-world use cases.

### On Chip RAM

On Chip RAM acts as a buffer between TCM and DDR. This memory can be shared by all the subsystems, including all the NPUs (*if the SoC has multiple NPUs*). To best utilize DDR bandwidth (as discussed in the previous section), it is beneficial to

1. Access DDR in big chunks. Helps more efficient reads and writes.
2. Increase data reuse to keep the tiles longer in the SoC internal memory. This will be clearer after the tiling chapter.
3. Share data (weights, intermediate results) between NPUs.

Thus, a well sized SRAM can reduce DDR traffic and make DDR access more optimal. However, SRAM has cost and area impact which imposes constraints on how big SRAM can be placed inside the SoC. Therefore, SRAM sizing is an important design decision affecting PPA. In section <TBD> we explore a HW-SW co-design methodology to come up with an optimal SRAM size for the desired system performance.

 

## Data Movement Engines

### CPU Load/Store Unit (LSU)

In a load-store architecture, CPU performs logic and math operations on the operands in the registers. LSU is responsible for 

1. Loading the operands from memory to CPU’s register files.
2. Writing the results from register files to memory.

In most modern CPUs ld/st instructions are pipelined to run concurrently with math and logic instructions, but they can still cause pipeline stalls for e.g. in case of a cache miss. 

### Vector LSU

VLSU is a fully pipelined LSU unit to move a block of data between memory and Vector Register File. It supports three types of addressing

1. Unit Stride – Fastest
2. Non-unit, but constant stride
3. Indexed (gather-scatter) – Slowest, but helpful for operations like table lookup etc. For e.g. activation functions like *tanh*, *sigmoid* etc. can be specified through a lookup table and executed on vector via indexed addressing. 

 

### Matrix LSU

Keeping the MAC arrays fed with data is a challenging task. The usual way of CPU’s LSU reading from cache will not be able to provide sufficient bandwidth and will often cause stalls because of cache misses. Therefore, a better solution is needed.

AI data has some special attributes, for e.g.

1. Multi-dimensional tensors, strided access. Not cache friendly.
2. Skip pattern reads.
3. Packed formats (e.g. two INT4 weight packed in one byte).
4. Memory access pattern is predictable. Therefore it is a good practice to bypass data caches. 

 

This creates the need for an AI specific data movement engine which can read a “tile” of data from memory and feed the MPU registers without involving the CPU. The tile is expected to be a small portion of a multi-dimensional tensor; hence the data movement engine must be capable of address generation using tensor dimensions.

 

RISC-V (proposed) matrix extension defines matrix load and store instructions but does not specify LSU behavior. For best performance it is critical that Matrix LSU can fetch the data directly from memory, offloading the CPU. It is also recommended that Matrix LSU reads/writes to a TCM or on-chip SRAM instead of DDR. How does the data move between DDR and TCM/SRAM? See the next section. 

 

It must be noted that for VLSU and MLSU to feed the execution units at the desired rate, Memory system must provide enough sustained bandwidth of (# lanes x SEW) /clock cycle.

 

### Direct Memory Access (DMA) Engines

A DMA engine can move data from almost anywhere to anywhere in the SoC, as long as source and destination are mapped in system memory map. In most high performance compute (e.g. multimedia, AI etc.), DMAs are commonly used to move inputs, outputs and parameters between DDR and SoC internal memories (e.g. SRAM). This is a faster, cheaper and power efficient method to offload the CPU and other processing elements from data movement. DMA can work in parallel with compute, which allows a pipelined architecture.

An example of pipelining is where a tensor is broken down into small tiles. At any given time:

1. Compute elements are processing tile #N, reading and writing to SRAM.
2. DMA is writing out the results of tile #N-1 from SRAM to DDR.
3. DMA is proactively fetching input(s) for tile #N+1 from DDR to SRAM.

If there is only one DMA, steps #2 and #3 may be done sequentially. The total execution time will be 

*max(t1, t2+t3)*

Therefore, to get best utilization of the precious compute elements, memory movement must be faster than compute. This is where DMAs excel.

 

## Bringing it all together

1. The full AI model (millions or billions of parameters), resides in DDR.
2. Every few milliseconds, a new input tensor like camera image is written to DDR.
3. SoC DMA moves a block of weights and activations from DDR to SRAM.
4. Vector/Matrix LSU moves smaller tiles from SRAM to register files.
5. Computations are done in the register space. 
6. Vector/Matrix LSU stores the results in SRAM. These results may be reused for next layer of processing. Therefore, the system tries to keep the intermediate results in SRAM as long as possible.
7. When SRAM is full, or the final results are available, SoC SMA writes a big block of data to DDR.

This is how multi-level memory hierarchy and data movement architecture results in most optimal performance, power and cost.