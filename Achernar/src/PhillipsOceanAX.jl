module PhillipsOceanAX

import Axis.Ocean as AXOcean

export RESOLUTION, DOMAIN_SIZE, COMPONENT_COUNT
export GRAVITY, WIND_SPEED, WIND_DIRECTION, AMPLITUDE_SCALE
export FRAME_BUFFER
export phillips_spectrum, compute_wave!, init!, update_params!

#=
Simulation parameters — Achernar overrides Axis.Ocean defaults.
=#
const RESOLUTION      = 512
const DOMAIN_SIZE     = 36.0f0
const COMPONENT_COUNT = 128

const GRAVITY         = 9.81f0
const WIND_SPEED      = 14.0f0
const WIND_DIRECTION  = (0.92f0, 0.38f0)
const AMPLITUDE_SCALE = 0.08f0

# Output buffer ( pre-allocated once, zero GC in hot path )
const FRAME_BUFFER = Vector{Float32}(undef, RESOLUTION * RESOLUTION)

# Holds the active simulation instance
const _SIM = Ref{Union{Nothing, AXOcean.PhillipsSim}}(nothing)

#=
Public API
=#
phillips_spectrum(kx::Real, ky::Real, windx::Real, windy::Real) =
    AXOcean.phillips_spectrum(Float32(kx), Float32(ky), Float32(windx), Float32(windy))

function compute_wave!(data::Vector{Float32}, t::Real)
    AXOcean.compute_wave!(_SIM[], data, t)
end

compute_wave!(t::Real) = compute_wave!(FRAME_BUFFER, t)

"""
    update_params!(; wind_speed=nothing, wind_direction=nothing, amplitude_scale=nothing)

Dynamic update API for Achernar. Triggers Axis.Ocean parameters update
without reallocating memory or touching the rust FFI lifecycle unnecessarily.
"""
function update_params!(; wind_speed=nothing, wind_direction=nothing, amplitude_scale=nothing)
    sim = _SIM[]
    sim === nothing && return
    AXOcean.update_params!(sim; wind_speed, wind_direction, amplitude_scale)
end

function init!()
    sim = AXOcean.create_phillips_sim(
        resolution      = RESOLUTION,
        component_count = COMPONENT_COUNT,
        domain_size     = DOMAIN_SIZE,
        gravity         = GRAVITY,
        wind_speed      = WIND_SPEED,
        wind_direction  = WIND_DIRECTION,
        amplitude_scale = AMPLITUDE_SCALE,
        seed            = 42,
    )
    AXOcean.init!(sim)
    _SIM[] = sim
    @info "PhillipsOcean Initialized" backend="wgpu ( Axis )"
end

end # module PhillipsOceanAX