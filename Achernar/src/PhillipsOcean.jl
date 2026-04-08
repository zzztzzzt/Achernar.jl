using Oxygen
using HTTP
using Random
using CUDA
using Base.Threads
using HTTP.WebSockets: WebSocketError, send

const RESOLUTION = 96
const FRAME_INTERVAL = 1 / 30
const DOMAIN_SIZE = 36.0f0
const COMPONENT_COUNT = 128

const GRAVITY = 9.81f0
const WIND_SPEED = 14.0f0
const WIND_DIRECTION = (0.92f0, 0.38f0)
const AMPLITUDE_SCALE = 0.08f0

const CUDA_WAVE_ENABLED = let available = false
    try
        available = CUDA.functional()
    catch
        available = false
    end
    available
end
const ENABLE_THREADED_WAVE = nthreads() > 1

#=
Precomputed Storage (SoA layout)
=#

const KX = Vector{Float32}(undef, COMPONENT_COUNT)
const KY = Vector{Float32}(undef, COMPONENT_COUNT)
const OMEGA = Vector{Float32}(undef, COMPONENT_COUNT)
const AMP = Vector{Float32}(undef, COMPONENT_COUNT)
const PHASE0 = Vector{Float32}(undef, COMPONENT_COUNT)

# Spatial phase cache:
# PHASE_BASE[i, j] = kx[j] * x[i] + ky[j] * y[i]
const PHASE_BASE = Matrix{Float32}(undef, RESOLUTION * RESOLUTION, COMPONENT_COUNT)

# Frame buffer ( reused every frame, zero allocation )
const FRAME_BUFFER = Vector{Float32}(undef, RESOLUTION * RESOLUTION)
const d_PHASE_BASE = CUDA_WAVE_ENABLED ? CuArray(PHASE_BASE) : nothing
const d_OMEGA = CUDA_WAVE_ENABLED ? CuArray(OMEGA) : nothing
const d_AMP = CUDA_WAVE_ENABLED ? CuArray(AMP) : nothing
const d_PHASE0 = CUDA_WAVE_ENABLED ? CuArray(PHASE0) : nothing
const d_FRAME_BUFFER = CUDA_WAVE_ENABLED ? CUDA.zeros(Float32, RESOLUTION * RESOLUTION) : nothing

# Grid coordinates
const GRID_X = Float32[
    ((x - 1) / (RESOLUTION - 1) - 0.5f0) * DOMAIN_SIZE
    for _ in 1:RESOLUTION, x in 1:RESOLUTION
]

const GRID_Y = Float32[
    ((y - 1) / (RESOLUTION - 1) - 0.5f0) * DOMAIN_SIZE
    for y in 1:RESOLUTION, _ in 1:RESOLUTION
]

#=
Utility Functions
=#

"""
Normalize a 2D vector safely.
Returns (0,0) if the input length is near zero.
"""
function normalize2(x::Float32, y::Float32)
    len = sqrt(x * x + y * y)
    if len < 1f-6
        return (0f0, 0f0)
    end
    return (x / len, y / len)
end

"""
Phillips spectrum for ocean wave energy distribution.
This models how wind transfers energy into waves.
"""
function phillips_spectrum(kx::Float32, ky::Float32, windx::Float32, windy::Float32)
    k2 = kx * kx + ky * ky
    if k2 < 1.0f-6
        return 0.0f0
    end

    k = sqrt(k2)
    k_unit_x = kx / k
    k_unit_y = ky / k

    alignment = max(k_unit_x * windx + k_unit_y * windy, 0.0f0)

    L = (WIND_SPEED * WIND_SPEED) / GRAVITY
    damping = 0.0015f0
    l2 = (L * damping)^2

    return exp(-1.0f0 / (k2 * L * L)) / (k2 * k2) *
           alignment^4 *
           exp(-k2 * l2)
end

#=
Initialization
=#

"""
Build wave components using a spectral model.
Uses Structure-of-Arrays layout for cache efficiency.
"""
function build_components!()
    rng = MersenneTwister(42)
    windx, windy = normalize2(WIND_DIRECTION...)

    pair_count = div(COMPONENT_COUNT, 2)

    idx = 1
    for i in 1:pair_count
        band = Float32(i - 1) / Float32(pair_count - 1)
        wavelength = 1.2f0 + 9.0f0 * band^2
        k = Float32(2 * pi) / wavelength

        spread = 1.05f0
        angle = Float32(atan(windy, windx)) +
                randn(rng, Float32) * spread * (0.2f0 + 0.8f0 * band)

        kx_main = cos(angle) * k
        ky_main = sin(angle) * k

        spectrum_main = phillips_spectrum(kx_main, ky_main, windx, windy)

        AMP[idx] = AMPLITUDE_SCALE *
                   sqrt(max(spectrum_main, 0f0)) *
                   (0.35f0 + 0.65f0 * (1f0 - band))

        PHASE0[idx] = rand(rng, Float32) * Float32(2pi)
        OMEGA[idx] = sqrt(GRAVITY * k)
        KX[idx] = kx_main
        KY[idx] = ky_main

        idx += 1

        # backward wave
        kx_back = -kx_main
        ky_back = -ky_main

        spectrum_back = phillips_spectrum(kx_back, ky_back, windx, windy)

        AMP[idx] = AMPLITUDE_SCALE * 0.45f0 *
                   sqrt(max(spectrum_back, 0f0) + max(spectrum_main, 0f0) * 0.25f0) *
                   (0.35f0 + 0.65f0 * (1f0 - band))

        PHASE0[idx] = rand(rng, Float32) * Float32(2pi)
        OMEGA[idx] = sqrt(GRAVITY * k)
        KX[idx] = kx_back
        KY[idx] = ky_back

        idx += 1
    end
end

"""
Precompute spatial phase :
kx*x + ky*y for every (pixel, component)
"""
function precompute_phase!()
    @inbounds for i in eachindex(GRID_X)
        x = GRID_X[i]
        y = GRID_Y[i]

        for j in 1:COMPONENT_COUNT
            PHASE_BASE[i, j] = KX[j] * x + KY[j] * y
        end
    end
end

function upload_wave_constants!()
    CUDA_WAVE_ENABLED || return
    copyto!(d_PHASE_BASE, PHASE_BASE)
    copyto!(d_OMEGA, OMEGA)
    copyto!(d_AMP, AMP)
    copyto!(d_PHASE0, PHASE0)
end

build_components!()
precompute_phase!()
upload_wave_constants!()

#=
Wave Simulation
=#

function wave_kernel!(frame, phase_base, omega, amp, phase0, tf)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx <= length(frame)
        h = 0.0f0

        @inbounds for j in 1:COMPONENT_COUNT
            phase = phase_base[idx, j] - omega[j] * tf + phase0[j]
            h += amp[j] * cos(phase)
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
            phase = PHASE_BASE[i, j] - OMEGA[j] * tf + PHASE0[j]
            @fastmath h += AMP[j] * cos(phase)
        end

        data[i] = h
    end
end

function compute_wave_threaded!(data::Vector{Float32}, t::Float64)
    tf = Float32(t)

    Threads.@threads for i in eachindex(data)
        h = 0.0f0

        @inbounds @simd for j in 1:COMPONENT_COUNT
            phase = PHASE_BASE[i, j] - OMEGA[j] * tf + PHASE0[j]
            @fastmath h += AMP[j] * cos(phase)
        end

        data[i] = h
    end
end

function compute_wave_gpu!(data::Vector{Float32}, t::Float64)
    threads = 256
    blocks = cld(length(data), threads)

    @cuda threads=threads blocks=blocks wave_kernel!(d_FRAME_BUFFER, d_PHASE_BASE, d_OMEGA, d_AMP, d_PHASE0, Float32(t))
    CUDA.synchronize()
    copyto!(data, d_FRAME_BUFFER)
end

function compute_wave!(data::Vector{Float32}, t::Float64)
    if CUDA_WAVE_ENABLED
        compute_wave_gpu!(data, t)
    elseif ENABLE_THREADED_WAVE
        compute_wave_threaded!(data, t)
    else
        compute_wave_serial!(data, t)
    end
end

#=
WebSocket Stream
=#

@websocket "/phillips-ocean" function(ws::HTTP.WebSocket)
    start_time = time()

    try
        while true
            frame_start = time()

            t = frame_start - start_time

            compute_wave!(FRAME_BUFFER, t)

            # Zero-copy reinterpretation
            payload = reinterpret(UInt8, FRAME_BUFFER)
            send(ws, payload)

            # Frame pacing ( stable FPS )
            elapsed = time() - frame_start
            sleep(max(0, FRAME_INTERVAL - elapsed))
        end
    catch err
        if !isa(err, WebSocketError)
            rethrow(err)
        end
    end
end

println(CUDA_WAVE_ENABLED ? "Wave compute backend: CUDA.jl" : "Wave compute backend: CPU")

serve()
