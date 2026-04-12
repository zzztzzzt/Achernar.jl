"""
PhillipsOceanWS

WebSocket streaming server for the Phillips Ocean wave simulator.
Depends on `PhillipsOcean.jl` for all physics computation.

Usage

```julia
julia --project=. --threads=auto src/PhillipsOceanWS.jl
```

Endpoint : ws://localhost:8080/phillips-ocean

Each message is a binary frame of `RESOLUTION x RESOLUTION` little-endian
`Float32` values ( ~36 KB ), pushed at ~30 fps. Client-side access :

```javascript
const heights = new Float32Array(event.data); // heights[row * 96 + col]
```
"""

using Oxygen
using HTTP
using HTTP.WebSockets: WebSocketError, send

include("PhillipsOcean.jl")
using .PhillipsOcean

#=
WebSocket Route
=#

@websocket "/phillips-ocean" function(ws::HTTP.WebSocket)
    start_time = time()

    try
        while true
            frame_start = time()

            t = frame_start - start_time

            compute_wave!(FRAME_BUFFER, t)

            # Zero-copy reinterpretation of Float32 buffer as raw bytes
            payload = reinterpret(UInt8, FRAME_BUFFER)
            send(ws, payload)

            # Frame pacing – maintain stable ~30 FPS
            elapsed = time() - frame_start
            sleep(max(0, FRAME_INTERVAL - elapsed))
        end
    catch err
        if !isa(err, WebSocketError)
            rethrow(err)
        end
    end
end

#=
Entrypoint
=#

serve()
