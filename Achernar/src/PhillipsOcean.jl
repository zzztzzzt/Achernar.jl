"""
PhillipsOcean
Computes ocean height via Phillips Spectrum across CUDA, Multi-threaded, or Serial backends.

Quick Start :
1. Setup : `include("PhillipsOcean.jl"); using .PhillipsOcean`
2. Init : `init!()` is called automatically on load to configure GPU/CPU resources.
3. Run : `compute_wave!(FRAME_BUFFER, t)` inside your loop.
        - `t` : Elapsed time in seconds ( Float64 ).
        - `FRAME_BUFFER` : Vector{Float32} mapped to 2D grid ( Column-major ).

Notes :
- Configuration : Adjust `const` values ( WIND, AMP, etc. ) in-file and re-include.
- Performance : Run Julia with `--threads auto` for multi-threaded CPU support.
"""

module PhillipsOcean

using Random
using CUDA
using Base.Threads

#=
Public API
=#
export RESOLUTION, FRAME_INTERVAL, DOMAIN_SIZE, COMPONENT_COUNT
export GRAVITY, WIND_SPEED, WIND_DIRECTION, AMPLITUDE_SCALE
export FRAME_BUFFER
export normalize2, phillips_spectrum
export compute_wave!
export init!

#=
Constants
=#

const RESOLUTION = 96
const FRAME_INTERVAL = 1 / 30
const DOMAIN_SIZE = 36.0f0
const COMPONENT_COUNT = 128

const GRAVITY = 9.81f0
const WIND_SPEED = 14.0f0
const WIND_DIRECTION = (0.92f0, 0.38f0)
const AMPLITUDE_SCALE = 0.08f0

# Use Ref to store execution-time state to improve pre-compilation compatibility
const CUDA_WAVE_ENABLED = Ref(false)
const ENABLE_THREADED_WAVE = Ref(false)

#=
Precomputed Storage ( SoA layout )
=#
const KX = Vector{Float32}(undef, COMPONENT_COUNT)
const KY = Vector{Float32}(undef, COMPONENT_COUNT)
const OMEGA = Vector{Float32}(undef, COMPONENT_COUNT)
const AMP = Vector{Float32}(undef, COMPONENT_COUNT)
const PHASE0 = Vector{Float32}(undef, COMPONENT_COUNT)
const PHASE_BASE = Matrix{Float32}(undef, RESOLUTION * RESOLUTION, COMPONENT_COUNT)
const FRAME_BUFFER = Vector{Float32}(undef, RESOLUTION * RESOLUTION)

# Grid coordinates
const GRID_X = Float32[((x - 1) / (RESOLUTION - 1) - 0.5f0) * DOMAIN_SIZE for _ in 1:RESOLUTION, x in 1:RESOLUTION]
const GRID_Y = Float32[((y - 1) / (RESOLUTION - 1) - 0.5f0) * DOMAIN_SIZE for y in 1:RESOLUTION, _ in 1:RESOLUTION]

# GPU mirrors buffer ( using Ref{Any} to avoid attempting to allocate GPU memory during pre-compilation )
const d_PHASE_BASE = Ref{Any}(nothing)
const d_OMEGA = Ref{Any}(nothing)
const d_AMP = Ref{Any}(nothing)
const d_PHASE0 = Ref{Any}(nothing)
const d_FRAME_BUFFER = Ref{Any}(nothing)

#=
Utility Functions
=#
function normalize2(x::Float32, y::Float32)
    len = sqrt(x * x + y * y)
    return len < 1f-6 ? (0f0, 0f0) : (x / len, y / len)
end

function phillips_spectrum(kx::Float32, ky::Float32, windx::Float32, windy::Float32)
    k2 = kx * kx + ky * ky
    k2 < 1.0f-6 && return 0.0f0
    k = sqrt(k2)
    alignment = max((kx / k) * windx + (ky / k) * windy, 0.0f0)
    L = (WIND_SPEED * WIND_SPEED) / GRAVITY
    l2 = (L * 0.0015f0)^2
    return exp(-1.0f0 / (k2 * L * L)) / (k2 * k2) * alignment^4 * exp(-k2 * l2)
end

#=
Initialization Logic
=#
function build_components!()
    rng = MersenneTwister(42)
    windx, windy = normalize2(WIND_DIRECTION...)
    pair_count = div(COMPONENT_COUNT, 2)
    idx = 1
    for i in 1:pair_count
        band = Float32(i - 1) / Float32(pair_count - 1)
        wavelength = 1.2f0 + 9.0f0 * band^2
        k = Float32(2 * pi) / wavelength
        angle = Float32(atan(windy, windx)) + randn(rng, Float32) * 1.05f0 * (0.2f0 + 0.8f0 * band)
        
        # Forward & Backward wave setup
        for (dir, scale) in [(1.0f0, 1.0f0), (-1.0f0, 0.45f0)]
            kx, ky = dir .* (cos(angle) * k, sin(angle) * k)
            spec = phillips_spectrum(kx, ky, windx, windy)
            AMP[idx] = AMPLITUDE_SCALE * scale * sqrt(max(spec, 0f0)) * (0.35f0 + 0.65f0 * (1f0 - band))
            PHASE0[idx] = rand(rng, Float32) * Float32(2pi)
            OMEGA[idx] = sqrt(GRAVITY * k)
            KX[idx], KY[idx] = kx, ky
            idx += 1
        end
    end
end

function precompute_phase!()
    @inbounds for i in eachindex(GRID_X)
        for j in 1:COMPONENT_COUNT
            PHASE_BASE[i, j] = KX[j] * GRID_X[i] + KY[j] * GRID_Y[i]
        end
    end
end

"""
init!()
Responsible for performing environmental detection and GPU resource uploading
"""
function init!()
    CUDA_WAVE_ENABLED[] = CUDA.functional()
    ENABLE_THREADED_WAVE[] = nthreads() > 1

    if CUDA_WAVE_ENABLED[]
        d_PHASE_BASE[] = CuArray(PHASE_BASE)
        d_OMEGA[] = CuArray(OMEGA)
        d_AMP[] = CuArray(AMP)
        d_PHASE0[] = CuArray(PHASE0)
        d_FRAME_BUFFER[] = CUDA.zeros(Float32, RESOLUTION * RESOLUTION)
    end
    
    @info "PhillipsOcean Initialized" backend=(CUDA_WAVE_ENABLED[] ? "CUDA" : (ENABLE_THREADED_WAVE[] ? "CPU-threaded" : "CPU-serial"))
end

function __init__()
    init!() # Automatically perform hardware initialization each time it is used
end

#=
Wave Simulation Backends
=#
function wave_kernel!(frame, phase_base, omega, amp, phase0, tf)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(frame)
        h = 0.0f0
        @inbounds for j in 1:COMPONENT_COUNT
            h += amp[j] * cos(phase_base[idx, j] - omega[j] * tf + phase0[j])
        end
        frame[idx] = h
    end
    return nothing
end

function compute_wave_serial!(data::Vector{Float32}, t::Float64)
    tf = Float32(t)
    @inbounds for i in eachindex(data)
        h = 0.0f0
        @simd for j in 1:COMPONENT_COUNT
            @fastmath h += AMP[j] * cos(PHASE_BASE[i, j] - OMEGA[j] * tf + PHASE0[j])
        end
        data[i] = h
    end
end

function compute_wave_threaded!(data::Vector{Float32}, t::Float64)
    tf = Float32(t)
    Threads.@threads for i in eachindex(data)
        h = 0.0f0
        @inbounds @simd for j in 1:COMPONENT_COUNT
            @fastmath h += AMP[j] * cos(PHASE_BASE[i, j] - OMEGA[j] * tf + PHASE0[j])
        end
        data[i] = h
    end
end

function compute_wave_gpu!(data::Vector{Float32}, t::Float64)
    threads = 256
    blocks  = cld(length(data), threads)
    @cuda threads=threads blocks=blocks wave_kernel!(
        d_FRAME_BUFFER[], d_PHASE_BASE[], d_OMEGA[], d_AMP[], d_PHASE0[], Float32(t)
    )
    CUDA.synchronize()
    copyto!(data, d_FRAME_BUFFER[])
end

function compute_wave!(data::Vector{Float32}, t::Float64)
    if CUDA_WAVE_ENABLED[]
        compute_wave_gpu!(data, t)
    elseif ENABLE_THREADED_WAVE[]
        compute_wave_threaded!(data, t)
    else
        compute_wave_serial!(data, t)
    end
end

# CPU data is pre-calculated during the compilation phase
build_components!()
precompute_phase!()

end # module PhillipsOcean