# Introduction

## Scope & Purpose of this book

This book talks about performance architecture of **Embedded Inferencing NPUs for Edge AI SoCs**. We will take a holistic system view which will include hardware computation blocks, memory subsystem and of course software. While there are many hardware architecture options, we will assume a RISC-V based NPU for discussing architecture tradeoffs and performance analysis. The primary reason for that choice is the open nature of RISC-V which allows us to discuss it in an open forum like this. However, the contents of this book will be applicable to a wide range of matrix accelerator based NPUs (for e.g. [Texas Instruments C7 MMA](https://forum.digikey.com/t/ti-edge-ai-am6xa-processors-with-deep-learning-accelerators-and-its-efficiency/39813), [Qualcomm's Hexagon NPU](https://www.qualcomm.com/products/technology/processors/hexagon), [NVidia DLA](https://developer.nvidia.com/deep-learning-accelerator) and many more)

This book is organized into several chapters. In every chapter we will take one topic and explain it in an easy to understand way. Over the course of the next few chapters, we will have learned about 

1. Challenges and key careabouts of Embedded EdgeAI Inference.
2. RISC-V NPU hardware architecture and how it interacts with the rest of the SoC.
3. Best practices to get maximum performance at lowest power.
4. Software’s role in achieving the best possible performance from the hardware.

 

We understand there are numerous other alternative NPU architectures for AI inferencing. To name a few

- GPU
- In-Memory Compute
- Quantum Compute
- Neuromorphic Compute 

This book will not include them, but maybe a future one will. 



## Why are we writing this book

While the internet is full of blogs, e-books, papers and videos on AI, we have felt that Embedded AI Inferencing is a topic covered very sparsely. This also applies to courses taught at universities and online education platforms. Most of the online literature focusses on training and/or GPU based architectures. At the same time we believe that low power, low latency inferencing on the edge will be the fastest growing area. Enabling this ecosystem will need a new generation of engineers who have deep knowledge of how the math and logic of AI is brought to life on low cost, small form factor hardware running on

- The cellphone in your pocket
- Or the car you are driving
- Or the watch you are wearing
- Or any other AI enabled device around you

Unfortunately this knowledge is limited to a few practitioners who have learned it the hard way trying to optimize AI models on embedded NPUs. With this book, our goal is to disseminate this knowledge and also get feedback from other experts in the field.

 

## Target Audience

This will be a great resource if you are

- A student of AI algorithms and computer architecture.

- A practicing engineer, trying to dive deep into Edge AI.

- A systems or SoC architect, with interest in PPA (power, performance, area) optimization of your IP/SoC.
- A SW architect, trying to write AI inferencing software which gets the best performance from the given hardware.
- Or you are just curious about Edge AI and RISC-V.




It is expected that the reader is already familiar with basic concepts of AI/ML and fundamentals of computer architecture. Knowledge of RISC-V architecture will be helpful, but not mandatory. There are some links in the references section which maybe good resources for deeper dives into these topics.

##  

## What will you get by reading this

We know you are busy and there is a ton of content on AI already available. This book is designed to be unique in the sense that it will go deep into performance optimization of AI inferencing on resource constrained embedded SoCs. By the time you are done reading this book, you would have developed a good understanding of

- AI inference performance bottlenecks on embedded SoCs.
- Hardware design tradeoffs like Matmul Array Size, Vector Length, External Memory bandwidth, Internal SRAM size etc.
- AI model tradeoffs like pruning, quantization, layer fusion and hardware specific graph modifications.
- Best practices and software defined data-flows to achieve the most juice from the hardware.



The rest of this book is organized in the following chapters

1. [Chapter 1 - Hardware Architecture](Chapter 1 - Hardware Architecture.html)
2. [Chapter 2 - Inference vs Training](Chapter 2 - Inference vs Training.html)
3. [Chapter 3 - WTF is “TOPS](Chapter 3 - WTF is “TOPS.html)
4. [Chapter 4 - Common Neural Net Structures in Edge AI](Chapter 4 - Common Neural Net Structures in Edge AI.html)
5. [Chapter 5 - Data Types and impact on performance](Chapter 5 - Data Types and impact on performance.html)
6. [Chapter 6 - Tensor Layouts and impact on performance](Chapter 6 - Tensor Layouts and impact on performance.html)
7. [Chapter 7 - How are NN operations accelerated on NPU](Chapter 7 - How are NN operations accelerated on NPU.html)
   1. [Chapter 7.1 - GeMM](Chapter 7.1 - GeMM.html)
   2. [Chapter 7.2 - Convolutions](Chapter 7.2 - Convolutions.html)
   3. [Chapter 7.3 - Activations](Chapter 7.3 - Activations.html)
   4. [Chapter 7.4 -Normalization](Chapter 7.4 -Normalization.html)
   5. [Chapter 7.5 - GeMV](Chapter 7.5 - GeMV.html)
   6. [Chapter 7.6 - Attention](Chapter 7.6 - Attention.html)
8. [Chapter 8 - SoC Level Performance](Chapter 8 - SoC Level Performance.html)







