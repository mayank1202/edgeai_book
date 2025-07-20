# Chapter 2 - **0 Inference vs Training**

[TOC]



## **Key differences**

AI Inference is the process of running new (previously unseen) data through a pre-trained model to ‘infer’ what the model was trained to do. For e.g. a model to detect human faces in an image would have been trained on a dataset of positive and negative training images. In inference, the system takes a completely new image (e.g. from a live camera) and feeds it to the pre-trained model. The model will try to infer if the image contains any human faces, and the size/position of the faces found. From a computational complexity and system performance point of view, the key differences between training and inference are

1. Training has a backward and a forward pass. Inference has only **forward pass**.

 

2. Training has many many many epochs. In each epoch the model weights are updated, model outputs are computed and compared with the reference to evaluate the cost function which is to be minimized. In inference, you get only one chance to get it right. Therefore, inference computation complexity is lower, but timing and power budgets are much tighter.

 

3. Training is done on **large batches**. Batching can result in better utilization of compute elements and therefore higher throughput, but comes at the cost of increased latency. Inference usually requires low latency and therefore batches are small, commonly set = 1. This is a tradeoff which embedded system designers must be aware of.

 

4. **Data formats** – training is usually done on floating point data. The most important reason behind this is that most optimizers (like Gradient Descent, Adam etc.) need differentiable cost functions. Integer functions are not differentiable and therefore floating point is a better choice. For inference, we don’t have that constraint which makes it possible to quantize the weights and activations to fixed-point. Fixed point hardware is simpler and cheaper than floating-point. This allows designers to pack more multiply-add units in the same area and power budget. FP32 is the most commonly used format for training, but newer formats like BF16, FP8 etc. are gaining popularity. In inference, INT8 has been the most commonly used format, but INT4 is gaining traction and many inference NPUs now support INT4 format. More on that in chapter 5.



## **Why is edge inference important**

While training gets all the media attention, inference is really the end goal of AI. This is where AI solves real world problems like self-driving cars, speech recognition, health-monitoring and generating new content.

Why can’t all the inference run on cloud? A few reasons

1. **Latency** – For e.g. a self-driving car has only a few 100s of milliseconds to get an image from front camera, detect objects in its path and take a corrective action (braking, lane change etc.). It simply does not have the time to send the image to a cloud server and wait to get the results. All of the decision making must be done quickly and the only way to do it is by running inference on an onboard AI chip.

 

2. **Privacy** – If you are comfortable with the big tech collecting your voice, camera and financial data then cloud inference is for you. If you are as paranoid as I am, you need Edge Inference where your data never leaves your personal device.

 

3. **Cost** – Did you look at your AWS bill last month? What if you could invest in a device with a highly capable inference accelerator running your chosen models and you don’t have to make the Mag7 richer than they already are.

 

## **Why is edge inference challenging**

While model complexity is increasing and new frontier models are released every week, computational capacity is not increasing as quickly as before. [Jensen Huang says Moore's Law is dead](https://www.marketwatch.com/story/moores-laws-dead-nvidia-ceo-jensen-says-in-justifying-gaming-card-price-hike-11663798618)

​       How does it affect Edge AI inferencing? Let’s look at the constraints in an embedded SoC

1. **Limited Computational Resources** – Training a model is a one-time cost. Inference is an operational cost that is incurred with every request. To keep the cost down, inference chips have very tight area budgets. This impacts all the computational elements in the chip including CPU, Vector, Matrix and the internal memory subsystem. 

 

2. **Memory Bandwidth** – AI workloads have always been memory intensive. Let’s take example of a Resnet-50 with 224x224x3 input. This model generates <TBD> bytes of intermediate outputs per inference. To run this model @ 30 frames/second requires <TBD> of memory bandwidth. To make things worse, 

 

a. Input resolutions are increasing. Gone are the days of 224x224 (0.05 Mega Pixel) images. For e.g. a self-driving car uses Front Camera with a resolution somewhere between 5-8 Mega Pixels.

 

b. Newer models (like Transformer based) have a much lower [arithmetic intensity](https://www.iguazio.com/glossary/arithmetic-intensity/) than CNNs. What is arithmetic intensity? In simple words, it is the ratio of math operations vs memory access in a given workload. The impact of low arithmetic intensity is that the SoC spends more time moving data and there is a risk that the precious compute elements sit idle waiting to be fed.

 

c. Memory bandwidth is a **shared resource** in a SoC. Usually, there are higher priority masters on the bus which get precedence over NPU. An example is a camera capture device (e.g. CSI-2) writing to DDR, or a Display device reading from DDR. These are real-time masters which usually get priority over NPU. 

 

3. **Latency** – As we discussed in the previous section, keeping low latency is critical in many applications. This requires system designers to balance the competing priorities between throughput and latency to design a well-balanced system.

 

4. **Power** – Power is arguably the most important factor is designing an embedded system. 

 

a. Very often such systems are battery operated and SoC power draw is among the most significant contributors to **battery drain**.

 

b. **Thermal management** – High power results in heat dissipation which can damage the electronics, or reduce the lifetime. Therefore the devices are either actively or passively cooled. The system cost increases significantly with the cooling method. Liquid cooling removes heat the fastest but is also most expensive. Next comes air cooling (fans) followed by passive cooling (heat sinks). Designing a low power SoC helops achieve longer lifespan with simpler cooling system, resulting in very high cost savings. 

 

c. **PMIC cost** – Sometimes SoC need an external chip called PMIC (Power Management IC) to provide power. If the SoC needs more power (hence higher current) it may need multiple PMICs, or more expensive ones. Therefore, a low power SoC can result in PMIC cost savings as well.

 