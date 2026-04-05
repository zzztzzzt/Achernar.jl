using Oxygen
using HTTP
using Random
using HTTP.WebSockets: WebSocketError, send

const RESOLUTION = 96
const FRAME_INTERVAL = 1 / 30
const DOMAIN_SIZE = 18.0f0
const COMPONENT_COUNT = 128
const GRAVITY = 9.81f0
const WIND_SPEED = 14.0f0
const WIND_DIRECTION = (0.92f0, 0.38f0)
const AMPLITUDE_SCALE = 0.08f0

struct WaveComponent
    kx::Float32
    ky::Float32
    omega::Float32
    amplitude::Float32
    phase::Float32
end

function normalize2(x::Float32, y::Float32)
    len = sqrt(x * x + y * y)
    return (x / len, y / len)
end

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

    return exp(-1.0f0 / (k2 * L * L)) / (k2 * k2) * alignment^4 * exp(-k2 * l2)
end

function build_components()
    rng = MersenneTwister(42)
    windx, windy = normalize2(WIND_DIRECTION[1], WIND_DIRECTION[2])
    components = WaveComponent[]

    pair_count = COMPONENT_COUNT ÷ 2

    for i in 1:pair_count
        band = Float32(i - 1) / Float32(pair_count - 1)
        wavelength = 1.2f0 + 9.0f0 * band^2
        k = Float32(2 * pi) / wavelength

        spread = 1.05f0
        angle = Float32(atan(windy, windx)) + randn(rng, Float32) * spread * (0.2f0 + 0.8f0 * band)
        kx_main = Float32(cos(angle)) * k
        ky_main = Float32(sin(angle)) * k

        spectrum_main = phillips_spectrum(kx_main, ky_main, windx, windy)
        amplitude_main = AMPLITUDE_SCALE * sqrt(max(spectrum_main, 0.0f0)) * (0.35f0 + 0.65f0 * (1.0f0 - band))
        phase_main = rand(rng, Float32) * Float32(2 * pi)
        omega = Float32(sqrt(GRAVITY * k))

        push!(components, WaveComponent(kx_main, ky_main, omega, amplitude_main, phase_main))

        kx_back = -kx_main
        ky_back = -ky_main
        spectrum_back = phillips_spectrum(kx_back, ky_back, windx, windy)
        amplitude_back = AMPLITUDE_SCALE * 0.45f0 * sqrt(max(spectrum_back, 0.0f0) + max(spectrum_main, 0.0f0) * 0.25f0) * (0.35f0 + 0.65f0 * (1.0f0 - band))
        phase_back = rand(rng, Float32) * Float32(2 * pi)

        push!(components, WaveComponent(kx_back, ky_back, omega, amplitude_back, phase_back))
    end

    return components
end

const COMPONENTS = build_components()
const GRID_X = Float32[
    ((x - 1) / (RESOLUTION - 1) - 0.5f0) * DOMAIN_SIZE
    for _ in 1:RESOLUTION, x in 1:RESOLUTION
]
const GRID_Y = Float32[
    ((y - 1) / (RESOLUTION - 1) - 0.5f0) * DOMAIN_SIZE
    for y in 1:RESOLUTION, _ in 1:RESOLUTION
]

function compute_wave_data(t::Float64)
    data = Vector{Float32}(undef, RESOLUTION * RESOLUTION)
    tf = Float32(t)

    @inbounds for i in eachindex(data)
        x = GRID_X[i]
        y = GRID_Y[i]
        h = 0.0f0

        for component in COMPONENTS
            phase = component.kx * x + component.ky * y - component.omega * tf + component.phase
            h += component.amplitude * cos(phase)
        end

        data[i] = h
    end

    return data
end

function encode_wave_frame(data::Vector{Float32})
    return copy(reinterpret(UInt8, data))
end

@websocket "/wave" function(ws::HTTP.WebSocket)
    start_time = time()

    try
        while true
            t = time() - start_time
            payload = encode_wave_frame(compute_wave_data(t))

            send(ws, payload)
            sleep(FRAME_INTERVAL)
        end
    catch err
        if !isa(err, WebSocketError)
            rethrow(err)
        end
    end
end

serve()
