module PhillipsOcean

using CUDA

# Public API
export RESOLUTION, FRAME_INTERVAL, DOMAIN_SIZE, COMPONENT_COUNT
export GRAVITY, WIND_SPEED, WIND_DIRECTION, AMPLITUDE_SCALE
export FRAME_BUFFER
export normalize2, phillips_spectrum
export compute_wave!
export init!

# Constants
const RESOLUTION = 512
const FRAME_INTERVAL = 1 / 60
const DOMAIN_SIZE = 36.0f0
const COMPONENT_COUNT = 128

const GRAVITY = 9.81f0
const WIND_SPEED = 14.0f0
const WIND_DIRECTION = (0.92f0, 0.38f0)
const AMPLITUDE_SCALE = 0.08f0

# Precomputed Storage ( SoA layout )
const KX = Vector{Float32}(undef, COMPONENT_COUNT)
const KY = Vector{Float32}(undef, COMPONENT_COUNT)
const OMEGA = Vector{Float32}(undef, COMPONENT_COUNT)
const AMP = Vector{Float32}(undef, COMPONENT_COUNT)
const PHASE0 = Vector{Float32}(undef, COMPONENT_COUNT)
const PHASE_BASE = Matrix{Float32}(undef, RESOLUTION * RESOLUTION, COMPONENT_COUNT)
const FRAME_BUFFER = Vector{Float32}(undef, RESOLUTION * RESOLUTION)

# Grid coordinates (x changes fast, y changes slow, matching WGSL)
const GRID_X = Float32[((x - 1) / (RESOLUTION - 1) - 0.5f0) * DOMAIN_SIZE for x in 1:RESOLUTION, y in 1:RESOLUTION]
const GRID_Y = Float32[((y - 1) / (RESOLUTION - 1) - 0.5f0) * DOMAIN_SIZE for x in 1:RESOLUTION, y in 1:RESOLUTION]

# GPU buffers ( using Ref{Any} to avoid attempting to allocate GPU memory during pre-compilation )
const d_PHASE_BASE = Ref{Any}(nothing)
const d_OMEGA = Ref{Any}(nothing)
const d_AMP = Ref{Any}(nothing)
const d_PHASE0 = Ref{Any}(nothing)
const d_FRAME_BUFFER = Ref{Any}(nothing)

# Utility Functions
function normalize2(x::Float32, y::Float32)
    len = sqrt(x * x + y * y)
    return len < 1f-6 ? (0f0, 0f0) : (x / len, y / len)
end

function phillips_spectrum(kx::Float32, ky::Float32, windx::Float32, windy::Float32)
    k2 = kx^2 + ky^2
    k2 == 0f0 && return 0f0
    k_dir_x, k_dir_y = normalize2(kx, ky)
    k_dot_w = k_dir_x * windx + k_dir_y * windy
    k_dot_w < 0f0 && return 0f0

    L = (WIND_SPEED^2) / GRAVITY
    return (exp(-1f0 / (k2 * L^2)) * (k_dot_w^2)) / (k2^2)
end

# Minimal xorshift64 RNG ( matches Rust AxisRng for identical numeric output )
mutable struct _AxisRng; state::UInt64; end
_AxisRng(seed::Integer) = _AxisRng(UInt64(max(seed, 1)))

function _next_u32!(r::_AxisRng)::UInt32
    x = r.state
    x ⊻= x >> 12; x ⊻= x << 25; x ⊻= x >> 27
    r.state = x
    UInt32((x * 0x2545f4914f6cdd1d % typemax(UInt64)) >> 32)
end

_next_f32!(r::_AxisRng)::Float32 =
    Float32(_next_u32!(r) >> 8) * (1f0 / Float32(1 << 24))

function _std_normal!(r::_AxisRng)::Float32
    u1 = max(_next_f32!(r), floatmin(Float32))
    sqrt(-2f0 * log(u1)) * cos(2f0 * Float32(π) * _next_f32!(r))
end

# Initialization Logic
function build_components!()
    rng = _AxisRng(42)
    windx, windy = normalize2(WIND_DIRECTION...)
    base_angle = atan(windy, windx)
    pair_count = div(COMPONENT_COUNT, 2)
    idx = 1
    for i in 0:(pair_count - 1)
        band = pair_count <= 1 ? 0f0 : Float32(i) / Float32(pair_count - 1)
        wavelength = 1.2f0 + 9.0f0 * band^2
        k = Float32(2 * pi) / wavelength
        angle = base_angle + _std_normal!(rng) * 1.05f0 * (0.2f0 + 0.8f0 * band)
        
        # Forward & Backward wave setup
        for (dir, scale) in [(1.0f0, 1.0f0), (-1.0f0, 0.45f0)]
            kx, ky = dir .* (cos(angle) * k, sin(angle) * k)
            spec = phillips_spectrum(kx, ky, windx, windy)
            AMP[idx] = AMPLITUDE_SCALE * scale * sqrt(max(spec, 0f0)) * (0.35f0 + 0.65f0 * (1f0 - band))
            PHASE0[idx] = _next_f32!(rng) * Float32(2pi)
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

function init!()
    CUDA.functional() || error(
        "PhillipsOcean requires a functional CUDA device."
    )

    d_PHASE_BASE[] = CuArray(PHASE_BASE)
    d_OMEGA[] = CuArray(OMEGA)
    d_AMP[] = CuArray(AMP)
    d_PHASE0[] = CuArray(PHASE0)
    d_FRAME_BUFFER[] = CUDA.zeros(Float32, RESOLUTION * RESOLUTION)

    @info "PhillipsOcean Initialized" backend="CUDA"
end


# CUDA Kernel & Compute
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

function compute_wave!(data::Vector{Float32}, t::Float64)
    threads = 256
    blocks = cld(length(data), threads)
    @cuda threads=threads blocks=blocks wave_kernel!(
        d_FRAME_BUFFER[], d_PHASE_BASE[], d_OMEGA[], d_AMP[], d_PHASE0[], Float32(t)
    )
    CUDA.synchronize()
    copyto!(data, d_FRAME_BUFFER[])
end

# CPU data is pre-calculated during the compilation phase
build_components!()
precompute_phase!()

end # module PhillipsOcean