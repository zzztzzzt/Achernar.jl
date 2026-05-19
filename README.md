# Achernar.jl

[![GitHub last commit](https://img.shields.io/github/last-commit/zzztzzzt/Achernar.jl.svg)](https://github.com/zzztzzzt/Achernar.jl)
[![GitHub repo size](https://img.shields.io/github/repo-size/zzztzzzt/Achernar.jl.svg)](https://github.com/zzztzzzt/Achernar.jl)

<br>

<img src="https://github.com/zzztzzzt/Achernar.jl/blob/main/logo/logo.png" alt="achernar-logo" style="height: 280px; width: auto;" />

### Achernar - 3D Physics Simulation in Three.js.

IMPORTANT : This project is still in the development and testing stages, licensing terms may be updated in the future. Please don't do any commercial usage currently.

## Project Dependencies Guide

[![Julia](https://img.shields.io/badge/Julia-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://github.com/JuliaLang/julia)
[![OxygenJl](https://img.shields.io/badge/Oxygen.jl-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://github.com/OxygenFramework/Oxygen.jl)
[![CUDAJl](https://img.shields.io/badge/CUDA.jl-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://github.com/juliagpu/cuda.jl)
[![WebGPU](https://img.shields.io/badge/WebGPU-005A9C?style=for-the-badge&logo=webgpu&logoColor=white)](https://www.w3.org/TR/webgpu/)
[![three.js](https://img.shields.io/badge/Three.js-000000?style=for-the-badge&logo=three.js&logoColor=white)](https://github.com/mrdoob/three.js/)
[![Vite](https://img.shields.io/badge/Vite-9135FF?style=for-the-badge&logo=vite&logoColor=white)](https://github.com/vitejs/vite)

**[ for Dependencies Details please see the end of this README ]**

Achernar uses Oxygen.jl & CUDA.jl to ensure the fast & stable physics simulation data sending process. Oxygen.jl & CUDA.jl licensed under the MIT License.  

Achernar uses Three.js ( with WebGPU ) & Vite to build frontend 3D Viewer. Three.js & Vite licensed under the MIT License.

![1.0showcase](https://github.com/zzztzzzt/Achernar.jl/blob/main/showcase/Achernar1.0.webp)

## Start 3D Viewer

put below folders to project-root ( from Fomalhaut & Axis project ) :

`Axis`, `axis_rs`, `Fomalhaut`, `fomalhaut_rs`

use below command to start it

`julia`

`] activate .`

`instantiate`

and hit Ctrl + D

activate Julia-Rust FFI :

`julia --project=. scripts/generate_bridge.jl`

start server :

`julia --project=. --threads=auto scripts/phillips_ocean_server.jl`

at project root, open another CMD

`npm install`

`npx vite`

you will see 3D viewer after these steps

## Change GPU Base

you can choose "Oxygen.jl + CUDA.jl" or "Fomalhaut + WGPU"

for example, go to `scripts/phillips_ocean_server.jl`

switch between `Achernar.PhillipsOceanOxygen.start()` & `Achernar.PhillipsOceanFMHUT.start_server()`

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

and hit `backspace`

```shell
using CUDA
```
```shell
CUDA.set_runtime_version!(v"13.1")
```

### If your Custom pkg has new Dependencies :

If you add any new dependencies into local Achernar's `project.toml`, you need to run below command on `every environment` which is using local Achernar ( or it won't auto update )

`julia`

`] activate .`

`dev path/to/project_root/YourNewPkgName`

### Resolve pkg problems :

If your pkg problems still exist, try below command too : 

( it will check the package dependencies )

`julia`

`] activate .`

`resolve`

## Project Dependencies Details

Oxygen.jl License : [https://github.com/OxygenFramework/Oxygen.jl/blob/master/LICENSE.md](https://github.com/OxygenFramework/Oxygen.jl/blob/master/LICENSE.md)
<br>

CUDA.jl License : [https://github.com/JuliaGPU/CUDA.jl/blob/master/LICENSE.md](https://github.com/JuliaGPU/CUDA.jl/blob/master/LICENSE.md)
<br>

Three.js License : [https://github.com/mrdoob/three.js/blob/dev/LICENSE](https://github.com/mrdoob/three.js/blob/dev/LICENSE)
<br>

Vite License : [https://github.com/vitejs/vite/blob/main/LICENSE](https://github.com/vitejs/vite/blob/main/LICENSE)