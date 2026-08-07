### Cluster Setup Steps

For those AMD fans, you can still run the CUDA samples on the Faculty Cluster.

In the following steps, I will walk you through the process of setting up a cluster environment to run CUDA samples.

I highly recommend you to also read more about the cluster environment and how to use it:

* [Cluster Usage Guide](https://guides.upb.ro/docs/grid/apptainer-usage)
* [ASC Course Page](https://ocw.cs.pub.ro/courses/asc/laboratoare/01)


#### Instructions

1. Connect to the cluster using SSH. You can use the following command in your terminal:

```bash
ssh moodle_username@fep.grid.pub.ro
```

2. Inspect the resources available on the cluster using the `sinfo` command:

```bash
   [mihnea.mitrache@fep10 ~]$  sinfo -o "%10P %30N %8c %10m %20G %10a"
PARTITION  NODELIST                       CPUS     MEMORY     GRES                 AVAIL
dgxa100    dgxa100-ncit-wn[01-04]         256      2063510    gpu:tesla_a100:8     up
dgxh100    dgxh100-precis-wn[01-03]       224      1998908+   gpu:tesla_h100:8     up
h200       ucsc-precis-h200-wn[11-17]     256      2321395+   gpu:nvidia_h200:8    up
haswell*   haswell-wn[29-42]              32       127309     (null)               up
hd         xl675dg10-wn175                96       773225     gpu:tesla_a100:10    up
ml         sprmcrogpu-wn[140-141]         112      128224     gpu:tesla_a100:2     up
sprmcrogpu sprmcrogpu-wn13                64       515093     gpu:rtx_2080ti:8     up
ucsx       ucsx-ncit-gpu-wn100            64       257158     gpu:tesla_a100:3     up
xl         xl270-wn[161-162]              56       257138     gpu:tesla_p100:2     up

```

Of course, we need to use the nodes with GPUs, so we will focus on the `dgxa100`, `dgxh100`, `h200`, `hd`, `ml`, `sprmcrogpu`, and `ucsx` partitions.

3. Next, you have to get a cuda image from DockerHub. You have to do so because the cluster no longer 
has direct cuda toolkit installed on the nodes, and you have to get accoustomed to getting an enviroment with the dependencies you need. You can use the following command to pull the image:

```bash

[mihnea.mitrache@fep10 ~]$ apptainer pull docker://nvidia/cuda:12.6.0-devel-ubuntu22.04
INFO:    Converting OCI blobs to SIF format
INFO:    Starting build...
INFO:    Fetching OCI image...
2.3GiB / 2.3GiB [==================================================================================] 100 % 25.2 MiB/s 0s1.3GiB / 1.3GiB [==================================================================================] 100 % 25.2 MiB/s 0s28.2MiB / 28.2MiB [================================================================================] 100 % 25.2 MiB/s 0s86.8KiB / 86.8KiB [================================================================================] 100 % 25.2 MiB/s 0s54.6MiB / 54.6MiB [================================================================================] 100 % 25.2 MiB/s 0s4.4MiB / 4.4MiB [==================================================================================] 100 % 25.2 MiB/s 0sINFO:    Extracting OCI image...
INFO:    Inserting Apptainer configuration...
INFO:    Creating SIF file...
[==============================================>-----------------------------------------------------------] 44 % 19m47s
[==============================================>-----------------------------------------------------------] 44 % 19m50s
[=============================================================================================================] 100 % 0s
```

4. After entering the node with GPU support, you can run the following command to enter the container and have access to the CUDA toolkit:

```bash
[mihnea.mitrache@fep10 ~]$ srun -p dgxa100 --gres gpu:1 --pty /bin/bash
srun: job 231689 queued and waiting for resources
srun: job 231689 has been allocated resources
GpuFreq=control_disabled
[mihnea.mitrache@dgxa100-ncit-wn02 ~]$ apptainer run --nv docker://nvidia/cuda:12.6.0-devel-ubuntu22.04
INFO:    Using cached SIF image

==========
== CUDA ==
==========

CUDA Version 12.6.0

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.


Apptainer> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Jun_14_16:34:21_PDT_2024
Cuda compilation tools, release 12.6, V12.6.20
Build cuda_12.6.r12.6/compiler.34431801_0
 ```

5. You now have an environment where you can run CUDA samples. You can compile and run the samples as you would on a local Linux machine. Yes, Linux, you can now have fun with some makefiles.

Simple Makefile:

```makefile
NVCC ?= nvcc
CUDA_ARCH ?= sm_80
NVCCFLAGS ?= -O2 -arch=$(CUDA_ARCH)

all: firstExe secondExe thirdExe fourthExe

firstExe: src/1/CountingGFLOPS.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

secondExe: src/2/CudaProperties.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

thirdExe: src/3/MatrixMultiply.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

fourthExe: src/4/VariableSum.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f firstExe secondExe thirdExe fourthExe

.PHONY: all clean
```

6. Copy files to the cluster using `scp` command. You will not have a lot of editing capabilities on the cluster, so it is better to prepare your files locally and then copy them to the cluster. You can use the following command to copy files:

```bash

scp -r Cluster moodle_username@fep.grid.pub.ro:~/
```

7. You are all set! :)
```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Jun_14_16:34:21_PDT_2024
Cuda compilation tools, release 12.6, V12.6.20
Build cuda_12.6.r12.6/compiler.34431801_0
Apptainer> make
nvcc -O2 -arch=sm_80 src/1/CountingGFLOPS.cu -o firstExe
nvcc -O2 -arch=sm_80 src/2/CudaProperties.cu -o secondExe
nvcc -O2 -arch=sm_80 src/3/MatrixMultiply.cu -o thirdExe
nvcc -O2 -arch=sm_80 src/4/VariableSum.cu -o fourthExe
Apptainer> ./firstExe
Time: 6.54083 ms
GFLOPS: 2626.56
Apptainer>
```