module PhillipsOceanAX

import Axis as AX

#=
Public API
=#
export RESOLUTION, DOMAIN_SIZE, COMPONENT_COUNT
export GRAVITY, WIND_SPEED, WIND_DIRECTION, AMPLITUDE_SCALE
export FRAME_BUFFER
export normalize2, phillips_spectrum
export compute_wave!, init!

#=
Constants
=#
const RESOLUTION      = 512
const DOMAIN_SIZE     = 36.0f0
const COMPONENT_COUNT = 128

const GRAVITY         = 9.81f0
const WIND_SPEED      = 14.0f0
const WIND_DIRECTION  = (0.92f0, 0.38f0)
const AMPLITUDE_SCALE = 0.08f0

#=
GPU resource IDs ( fixed, single-simulation module )
=#
const _BUF_FRAME      = 1
const _BUF_COMPONENTS = 2
const _BUF_PARAMS     = 3
const _PIPELINE_ID    = 1
const _WORKGROUP_SIZE = 256

#=
Precomputed Storage ( SoA layout )
Mirrors PhillipsOcean.jl — pre-allocated once, zero GC in hot path.
=#
const KX     = Vector{Float32}(undef, COMPONENT_COUNT)
const KY     = Vector{Float32}(undef, COMPONENT_COUNT)
const OMEGA  = Vector{Float32}(undef, COMPONENT_COUNT)
const AMP    = Vector{Float32}(undef, COMPONENT_COUNT)
const PHASE0 = Vector{Float32}(undef, COMPONENT_COUNT)

const FRAME_BUFFER = Vector{Float32}(undef, RESOLUTION * RESOLUTION)

# Reusable frame-level scratch buffers ( avoids any per-frame allocation )
const _COMPONENTS_BUF = Vector{Float32}(undef, COMPONENT_COUNT * 4)
const _PARAMS_BUF     = Vector{UInt8}(undef, 16)

#=
WGSL compute shader ( Phillips Ocean kernel )
Identical to the one in the old Axis.Ocean reference implementation.
=#
const _OCEAN_WGSL = """
struct Params {
    resolution      : u32,
    component_count : u32,
    time            : f32,
    domain_size     : f32,
}

@group(0) @binding(0) var<storage, read_write> frame      : array<f32>;
@group(0) @binding(1) var<storage, read>       components : array<vec4<f32>>; // (kx, ky, amp, dynamic_phase)
@group(0) @binding(2) var<uniform>             params     : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    let res = params.resolution;
    if (idx >= res * res) { return; }

    let ix = idx % res;
    let iy = idx / res;
    let x = (f32(ix) / f32(res - 1u) - 0.5) * params.domain_size;
    let y = (f32(iy) / f32(res - 1u) - 0.5) * params.domain_size;

    var height = 0.0;
    for (var c = 0u; c < params.component_count; c = c + 1u) {
        let comp = components[c];
        let phase = comp.x * x + comp.y * y + comp.w;
        height = height + comp.z * cos(phase);
    }
    frame[idx] = height;
}
"""

#=
Utility Functions
=#
function normalize2(x::Float32, y::Float32)
    len = sqrt(x * x + y * y)
    return len < 1f-6 ? (0f0, 0f0) : (x / len, y / len)
end

function phillips_spectrum(kx::Float32, ky::Float32, windx::Float32, windy::Float32)
    k2 = kx^2 + ky^2
    k2 == 0f0 && return 0f0
    k         = sqrt(k2)
    alignment = max((kx / k) * windx + (ky / k) * windy, 0f0)
    L         = (WIND_SPEED * WIND_SPEED) / GRAVITY
    l2_small  = (L * 0.0015f0)^2
    exp(-1f0 / (k2 * L * L)) / (k2 * k2) * alignment^4 * exp(-k2 * l2_small)
end

#=
Minimal xorshift64 RNG ( matches Rust AxisRng for identical numeric output )
=#
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

#=
Initialization Logic
Same algorithm as PhillipsOcean.jl::build_components! — runs once at module
load time (pure CPU, no GPU dependency).
=#
function build_components!()
    rng = _AxisRng(42)
    windx, windy = normalize2(WIND_DIRECTION...)
    base_angle = atan(windy, windx)
    pair_count = div(COMPONENT_COUNT, 2)
    idx = 1
    @inbounds for i in 0:(pair_count - 1)
        band  = pair_count <= 1 ? 0f0 : Float32(i) / Float32(pair_count - 1)
        k     = 2f0 * Float32(π) / (1.2f0 + 9.0f0 * band^2)
        angle = base_angle + _std_normal!(rng) * 1.05f0 * (0.2f0 + 0.8f0 * band)

        for (dir, scale) in ((1f0, 1f0), (-1f0, 0.45f0))
            kx  = dir * cos(angle) * k
            ky  = dir * sin(angle) * k
            spec = phillips_spectrum(kx, ky, windx, windy)
            AMP[idx]    = AMPLITUDE_SCALE * scale * sqrt(max(spec, 0f0)) * (0.35f0 + 0.65f0 * (1f0 - band))
            PHASE0[idx] = _next_f32!(rng) * 2f0 * Float32(π)
            OMEGA[idx]  = sqrt(GRAVITY * k)
            KX[idx]     = kx
            KY[idx]     = ky
            idx += 1
        end
    end
end

#=
Rust Hot-Path Functions
The two functions below replace _pack_components! and _write_params! from the
old Axis.Ocean implementation. They run every frame and must be zero-alloc.
Calling @AX.rust_fn:
  - generates a ccall stub here in Julia (same signature, zero overhead)
  - registers the Rust source, grouped under this file's module name
    ( "phillips_ocean_ax" ), ready for AX.generate_bridge().
=#

#=
    _pack_components!(kx, ky, amp, phase0, omega, t, dest, cc)

Pack SoA wave-component arrays into the interleaved vec4<f32> layout expected
by the WGSL shader: [ kx, ky, amp, phase0 - omega*t ] per component.
Zero Julia allocation — operates entirely on raw pointers into pre-allocated buffers.
=#
@AX.rust_fn function _pack_components!(
    kx::Ptr{Float32}, ky::Ptr{Float32}, amp::Ptr{Float32},
    phase0::Ptr{Float32}, omega::Ptr{Float32},
    t::Float32, dest::Ptr{Float32}, cc::Int32,
)::Cvoid
    """
    let cc    = cc as usize;
    let kx    = unsafe { std::slice::from_raw_parts(kx,    cc) };
    let ky    = unsafe { std::slice::from_raw_parts(ky,    cc) };
    let amp   = unsafe { std::slice::from_raw_parts(amp,   cc) };
    let ph0   = unsafe { std::slice::from_raw_parts(phase0, cc) };
    let omega = unsafe { std::slice::from_raw_parts(omega,  cc) };
    let dest  = unsafe { std::slice::from_raw_parts_mut(dest, cc * 4) };
    for i in 0..cc {
        let b = i * 4;
        dest[b]     = kx[i];
        dest[b + 1] = ky[i];
        dest[b + 2] = amp[i];
        dest[b + 3] = ph0[i] - omega[i] * t;
    }
    """
end

#=
    _write_params!(dest, resolution, component_count, t, domain_size)

Pack the 16-byte WGSL uniform buffer in-place.
Zero Julia allocation — writes directly into the pre-allocated _PARAMS_BUF.
=#
@AX.rust_fn function _write_params!(
    dest::Ptr{UInt8},
    resolution::UInt32, component_count::UInt32,
    t::Float32, domain_size::Float32,
)::Cvoid
    """
    let dest = unsafe { std::slice::from_raw_parts_mut(dest, 16) };
    dest[0..4].copy_from_slice(&resolution.to_le_bytes());
    dest[4..8].copy_from_slice(&component_count.to_le_bytes());
    dest[8..12].copy_from_slice(&t.to_le_bytes());
    dest[12..16].copy_from_slice(&domain_size.to_le_bytes());
    """
end

#=
GPU Initialization
=#
function init!()
    AX.wgpu_init!()

    fc = RESOLUTION * RESOLUTION
    cc = COMPONENT_COUNT

    AX.wgpu_create_buffer!(_BUF_FRAME,      fc * 4,      AX.BINDING_STORAGE_READ_WRITE)
    AX.wgpu_create_buffer!(_BUF_COMPONENTS, cc * 4 * 4,  AX.BINDING_STORAGE_READ)
    AX.wgpu_create_buffer!(_BUF_PARAMS,     16,           AX.BINDING_UNIFORM)

    # Upload t=0 initial data
    _pack_components!(
        pointer(KX), pointer(KY), pointer(AMP),
        pointer(PHASE0), pointer(OMEGA),
        0f0, pointer(_COMPONENTS_BUF), Int32(cc),
    )
    _write_params!(
        pointer(_PARAMS_BUF),
        UInt32(RESOLUTION), UInt32(cc), 0f0, DOMAIN_SIZE,
    )
    AX.wgpu_write_buffer!(_BUF_COMPONENTS, _COMPONENTS_BUF)
    AX.wgpu_write_buffer!(_BUF_PARAMS,     _PARAMS_BUF)

    binding_flags = UInt32[
        AX.BINDING_STORAGE_READ_WRITE,
        AX.BINDING_STORAGE_READ,
        AX.BINDING_UNIFORM,
    ]
    AX.wgpu_create_compute_pipeline!(_PIPELINE_ID, _OCEAN_WGSL, "main", binding_flags)
    AX.wgpu_bind_buffers!(_PIPELINE_ID, [_BUF_FRAME, _BUF_COMPONENTS, _BUF_PARAMS])

    @info "PhillipsOcean Initialized" backend="wgpu ( Axis )"
end

#=
Per-frame Compute
=#

"""
    compute_wave!(data, t)

Dispatch the GPU ocean kernel for time `t`, readback into `data`.
No heap allocation in the hot path.
"""
function compute_wave!(data::Vector{Float32}, t::Float64)
    fc = RESOLUTION * RESOLUTION
    tf = Float32(t)

    _pack_components!(
        pointer(KX), pointer(KY), pointer(AMP),
        pointer(PHASE0), pointer(OMEGA),
        tf, pointer(_COMPONENTS_BUF), Int32(COMPONENT_COUNT),
    )
    _write_params!(
        pointer(_PARAMS_BUF),
        UInt32(RESOLUTION), UInt32(COMPONENT_COUNT), tf, DOMAIN_SIZE,
    )

    AX.wgpu_write_buffer!(_BUF_COMPONENTS, _COMPONENTS_BUF)
    AX.wgpu_write_buffer!(_BUF_PARAMS,     _PARAMS_BUF)
    AX.wgpu_dispatch!(_PIPELINE_ID; wg_x = cld(fc, _WORKGROUP_SIZE))
    AX.wgpu_read_buffer!(_BUF_FRAME, data)

    return data
end

"""
    compute_wave!(t)

Convenience overload — writes into `FRAME_BUFFER`.
"""
compute_wave!(t::Float64) = compute_wave!(FRAME_BUFFER, t)

# CPU component data is pre-calculated during module load ( pure CPU, no GPU needed )
build_components!()

end # module PhillipsOceanAX