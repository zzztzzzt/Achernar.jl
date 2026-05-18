module PhillipsOceanOxygen

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

            # Convert the Float32 buffer to a raw byte
            payload = reinterpret(UInt8, FRAME_BUFFER)
            
            # Pack with Envelope V1 format
            io = IOBuffer()
            write(io, UInt8(1)) # version = 1
            write(io, htol(UInt16(1))) # contentType = 1 (Float32 Tensor)
            write(io, htol(UInt16(0))) # flags = 0
            write(io, htol(UInt64(time_ns()))) # timestampNs
            write(io, htol(UInt32(sizeof(FRAME_BUFFER)))) # payloadLen
            write(io, payload)
            
            send(ws, take!(io))

            # Frame rate control : Maintain at approximately 120 FPS
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
    init!()
    
    @websocket "/phillips-ocean" ocean_stream_handler
    
    @info "Phillips Ocean WebSocket Server started on ws://localhost:8080/phillips-ocean"
    serve()
end

end # module PhillipsOceanOxygen

if abspath(PROGRAM_FILE) == @__FILE__
    PhillipsOceanOxygen.start()
end