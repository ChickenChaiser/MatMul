# Matrix multiplication

## Overview

Matrix multiplication with CUDA.
Conducted a research of the speed of execution between the GPU(CUDA) and CPU.

## Usage

Run executable program: MatMul\x64\Debug\MatMul.exe

## System configuration

| Name    | Values  |
|---------|---------|
| CPU     | Intel® Core™ i7-7700HQ CPU @ 2.80GHz (Turbo Boost  3.80 GHz) × 8 |
| RAM     | 16 GB DDR4 |
| GPU     | GeForce GTX 1060 6 GB |
| OS type | 64-bit  |

## Conclusion5752027013  

Results of reseach. "SIZE" is mean multiplier size.
Average time in seconds over 10 measurements.
Elements of matrix have float type.

| SIZE      |    CPU    | GPU (BLOCK SIZE = 32) | GPU (BLOCK SIZE = 16) |
|-----------|-----------|-----------------------|-----------------------|
|  512x512  |   1.0439  |      0.014783293      |      0.014657024      |
| 1024x1024 |  17.0471  |      0.092691044      |      0.088989181      |
| 2048x2048 | 242.6022  |      0.642520764      |      0.643258868      |
| 4096x4096 | 1860.239  |      5.124904785      |      5.078233301      |
