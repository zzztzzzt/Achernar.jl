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

module PhillipsOceanWS

using Oxygen
using HTTP
using HTTP.WebSockets: WebSocketError, send

include("PhillipsOcean.jl")
using .PhillipsOcean

#=
WebSocket streaming logic
=#
function ocean_stream_handler(ws)
    start_time = time()
    try
        while true
            frame_start = time()
            t = frame_start - start_time

            compute_wave!(FRAME_BUFFER, t)

            # Convert the Float32 buffer to a raw byte ( Zero-copy )
            payload = reinterpret(UInt8, FRAME_BUFFER)
            send(ws, payload)

            # Frame rate control : Maintain at approximately 30 FPS
            elapsed = time() - frame_start
            sleep(max(0, FRAME_INTERVAL - elapsed))
        end
    catch err
        if err isa InterruptException
            rethrow(err)
        end
        # Other errors ( such as Broken Pipe or WebSocketError ) usually indicate that the client has disconnected; simply end the process quietly
    end
end

function start()
    @websocket "/phillips-ocean" ocean_stream_handler
    
    @info "Phillips Ocean WebSocket Server started on ws://localhost:8080/phillips-ocean"
    serve()
end

end # module PhillipsOceanWS

if abspath(PROGRAM_FILE) == @__FILE__
    PhillipsOceanWS.start()
end