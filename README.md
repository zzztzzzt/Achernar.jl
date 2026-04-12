# Achernar.jl

[![GitHub last commit](https://img.shields.io/github/last-commit/zzztzzzt/Achernar.jl.svg)](https://github.com/zzztzzzt/Lyra-AI)
[![GitHub repo size](https://img.shields.io/github/repo-size/zzztzzzt/Achernar.jl.svg)](https://github.com/zzztzzzt/Lyra-AI)

<br>

<img src="https://github.com/zzztzzzt/Achernar.jl/blob/main/logo/logo.png" alt="lyra-logo" style="height: 280px; width: auto;" />

### Achernar - 3D Physics Simulation for Three.js.

IMPORTANT : This project is still in the development and testing stages, licensing terms may be updated in the future. Please don't do any commercial usage currently.

## Project Dependencies Guide

[![Julia](https://img.shields.io/badge/Julia-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://github.com/JuliaLang/julia)
[![OxygenJl](https://img.shields.io/badge/Oxygen.jl-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://github.com/OxygenFramework/Oxygen.jl)
[![CUDAJl](https://img.shields.io/badge/CUDA.jl-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://github.com/juliagpu/cuda.jl)
[![three.js](https://img.shields.io/badge/Three.js-000000?style=for-the-badge&logo=three.js&logoColor=white)](https://github.com/mrdoob/three.js/)
[![Vite](https://img.shields.io/badge/Vite-9135FF?style=for-the-badge&logo=vite&logoColor=white)](https://github.com/vitejs/vite)

**[ for Dependencies Details please see the end of this README ]**

Achernar uses Oxygen.jl & CUDA.jl to ensure the fast & stable physics simulation data sending process. Oxygen.jl & CUDA.jl licensed under the MIT License.  

Achernar uses Three.js & Vite to build frontend 3D Viewer. Three.js & Vite licensed under the MIT License.

![1.0showcase](https://github.com/zzztzzzt/Achernar.jl/blob/main/showcase/Achernar1.0.webp)

## WIP Project Achernar

use below command to start it

`julia`

`] activate .`

`instantiate`

and hit Ctrl + D

`julia --project=. src/Metaballs.jl`

open another CMD

`npm install`

`npx vite`

you will see 3D viewer after these step

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

## Project Dependencies Details

Oxygen.jl License : [https://github.com/OxygenFramework/Oxygen.jl/blob/master/LICENSE.md](https://github.com/OxygenFramework/Oxygen.jl/blob/master/LICENSE.md)
<br>

CUDA.jl License : [https://github.com/JuliaGPU/CUDA.jl/blob/master/LICENSE.md](https://github.com/JuliaGPU/CUDA.jl/blob/master/LICENSE.md)
<br>

Three.js License : [https://github.com/mrdoob/three.js/blob/dev/LICENSE](https://github.com/mrdoob/three.js/blob/dev/LICENSE)
<br>

Vite License : [https://github.com/vitejs/vite/blob/main/LICENSE](https://github.com/vitejs/vite/blob/main/LICENSE)