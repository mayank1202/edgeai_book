# Chapter 8 - **SoC Level Performance**

[TOC]



## **Recap - SoC Hardware Architecture**

<img src="C:\work\edgeai_book\SoC_BD.jpg" alt="SoC_BD" style="zoom:50%;" />

​														SoC Block Diagram



​	![Mem_hierarchy](C:\work\edgeai_book\Mem_hierarchy.png)

​														Memory Hierarchy



| **Memory Type**      | **Cost/byte** | **Speed** | **Energy/byte** |
| -------------------- | ------------- | --------- | --------------- |
| Registers            | Highest       | Fastest   | Lowest          |
| TCM/Cache (L1,L2,L3) | High          | Fast      | Low             |
| On Chip RAM/LLC      | Mid           | Mid       | Mid             |
| DDR                  | Low           | Slow      | High            |
| Storage              | Lowest        | Slowest   | Highest         |

​								speed vs cost vs data movement energy comparison of different types of memories



## **How does it impact performance**

We have seen the micro-kernel performance and how it is impacted by load/store bandwidth. Peak performance of  $2T^2$  ops/cycle  can be achieved only  if Matrix LSU can load operands at T elements per cycle. For the 16 INT8 TOPS NPU (T=64, F=2GHz), we need memory bandwidth of 128 Gigabytes/s. This is very high and is impractical to achieve if the matrices reside in external memory (DDR). Therefore, moving memory is a two-step process



1. Large blocks of memory are moved from DDR to on-chip RAM. The block size chosen can be bigger than MPU tile. This is done to achieve higher efficiency on DDR channels. Please see Appendix <TBD> on DDR b/w. This data movement is usually done with the help of DMA engines so that CPU can be offloaded.

   

2. MPU consumes this data tile-by-tile. As we have seen before, it will use matrix/vector LSU to move the tiles into register files. 

   

3. In the micro-kernel, we had pipelined LSU and MAC to achieve parallelization. Similar parallelization can also be achieved between DMA and MPU at system level. This is usually done through a software managed pipeline as follows

   

   

**Ramp-up** 

	1. DMA-In tile #0
	2. DMA-In tile #1



**Steady State** 	

	1. Process tile #i.
	2. DMA-In tile #i+1. This will be computed when MPU is done processing tile#i.
	3. DMA-out output tile #i-1. This was computed in previous iteration.

**Ramp-down** 

	1. DMA-Out tile #N-2
	2. DMA-Out tile #N-1



In steady state, MPU compute is running in parallel with Dma-In and DMA-out operations. Total DMA time must be less than MPU compute time to make sure MPU is never starved for data. DMA time depends on

1. External memory bandwidth
2. Bandwidth utilization (more on that in chapter 11)
3. Memory access pattern
4. Contention with other cores in the SoC



This calls for an SoC architecture which can guarantee the DDR bandwidth required by MPU to perform at its peak.

