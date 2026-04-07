# Achernar.jl

[![GitHub last commit](https://img.shields.io/github/last-commit/zzztzzzt/Achernar.jl.svg)](https://github.com/zzztzzzt/Lyra-AI)
[![GitHub repo size](https://img.shields.io/github/repo-size/zzztzzzt/Achernar.jl.svg)](https://github.com/zzztzzzt/Lyra-AI)

<br>

<img src="https://github.com/zzztzzzt/Achernar.jl/blob/main/logo/logo.png" alt="lyra-logo" style="height: 280px; width: auto;" />

### Achernar - 3D Physics Simulation for Three.js.

IMPORTANT : This project is still in the development and testing stages, licensing terms may be updated in the future. Please don't do any commercial usage currently.

## Project Detail / Debug

### CUDA version conflict solving :

if you met below error during CUDA.jl downloading, let's take CUDA 13.1.0 for example :

```shell
┌ Error: You are using CUDA 13.1.0, but CUDA.jl was precompiled for CUDA 13.2.0.
│ This is unexpected; please file an issue.
```

you must run below :

```shell
julia
```
```shell
] activate .
```
```shell
using CUDA
```
```shell
CUDA.set_runtime_version!(v"13.1")
```