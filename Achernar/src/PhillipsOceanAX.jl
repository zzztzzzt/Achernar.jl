module PhillipsOceanAX

import Axis as AX

export RESOLUTION, FRAME_INTERVAL, DOMAIN_SIZE, COMPONENT_COUNT
export GRAVITY, WIND_SPEED, WIND_DIRECTION, AMPLITUDE_SCALE
export FRAME_BUFFER
export phillips_spectrum, compute_wave!, init!

#=
Constants
=#
const RESOLUTION = 96
const FRAME_INTERVAL = 1 / 120
const DOMAIN_SIZE = 36.0f0
const COMPONENT_COUNT = 128

const GRAVITY = 9.81f0
const WIND_SPEED = 14.0f0
const WIND_DIRECTION = (0.92f0, 0.38f0)
const AMPLITUDE_SCALE = 0.08f0

#=
Precomputed Storage
=#
const KX           = Vector{Float32}(undef, COMPONENT_COUNT)
const KY           = Vector{Float32}(undef, COMPONENT_COUNT)
const OMEGA        = Vector{Float32}(undef, COMPONENT_COUNT)
const AMP          = Vector{Float32}(undef, COMPONENT_COUNT)
const PHASE0       = Vector{Float32}(undef, COMPONENT_COUNT)
const PHASE_BASE   = Matrix{Float32}(undef, RESOLUTION * RESOLUTION, COMPONENT_COUNT)
const FRAME_BUFFER = Vector{Float32}(undef, RESOLUTION * RESOLUTION)

const GRID_X = Float32[((x - 1) / (RESOLUTION - 1) - 0.5f0) * DOMAIN_SIZE for _ in 1:RESOLUTION, x in 1:RESOLUTION]
const GRID_Y = Float32[((y - 1) / (RESOLUTION - 1) - 0.5f0) * DOMAIN_SIZE for y in 1:RESOLUTION, _ in 1:RESOLUTION]

#=
API
=#
phillips_spectrum(kx::Real, ky::Real, windx::Real, windy::Real) =
    AX.phillips_spectrum(Float32(kx), Float32(ky), Float32(windx), Float32(windy))

compute_wave!(data::Vector{Float32}, t::Real) = AX.compute_wave!(data, Float32(t))
compute_wave!(t::Real) = compute_wave!(FRAME_BUFFER, t)

function init!()
    AX.build_components!(
        KX, KY, OMEGA, AMP, PHASE0;
        component_count = COMPONENT_COUNT,
        wind_direction  = WIND_DIRECTION,
        wind_speed      = WIND_SPEED,
        gravity         = GRAVITY,
        amplitude_scale = AMPLITUDE_SCALE,
        seed            = 42,
    )
    AX.precompute_phase!(PHASE_BASE, KX, KY; grid_x=GRID_X, grid_y=GRID_Y, component_count=COMPONENT_COUNT)
    AX.init!()
    # Re-upload with our local buffers to ensure GPU uses this module's data
    AX.upload_buffers!(PHASE_BASE, OMEGA, AMP, PHASE0; component_count=COMPONENT_COUNT)
    @info "PhillipsOcean Initialized" backend="wgpu ( Axis )"
end


end # module PhillipsOceanAX