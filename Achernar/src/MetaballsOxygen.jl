module MetaballsOxygen

using Oxygen
using HTTP
using HTTP.WebSockets: WebSocketError, send

include("Metaballs.jl")
using .Metaballs

#=
WebSocket streaming logic
=#
function metaballs_stream_handler(ws::HTTP.WebSocket)
    println("Metaballs client connected")
    start_time = time()

    try
        while true
            frame_start = time()
            elapsed_total = frame_start - start_time

            update_physics!(elapsed_total)
            compute_field!(FIELD_BUFFER[])
            vertex_float_count = build_mesh!(VERTEX_BUFFER[], NORMAL_BUFFER[], FIELD_BUFFER[])
            payload = build_payload!(PAYLOAD_BUFFER[], VERTEX_BUFFER[], NORMAL_BUFFER[], vertex_float_count)

            send(ws, payload)

            elapsed_frame = time() - frame_start
            sleep(max(0.0, FRAME_INTERVAL - elapsed_frame))
        end
    catch err
        if !isa(err, WebSocketError) && !(err isa IOError && err.code == Base.UV_ECANCELED)
            @error "Metaballs WS Error" exception = err
        end
    end
end

function start(; port::Int = 8080)
    #=
    Allocate CPU & GPU buffers now ( runtime ), not during precompilation
    to prevant dll problems
    =#
    init_buffers!()

    @websocket "/metaballs" metaballs_stream_handler

    @info "Metaballs WebSocket Server started" url="ws://localhost:$(port)/metaballs" backend=(Metaballs._CUDA_ENABLED[] ? "CUDA.jl" : (Metaballs.ENABLE_THREADED_FIELD ? "CPU (threaded)" : "CPU (serial)"))
    serve(port = port)
end

end # module MetaballsOxygen

if abspath(PROGRAM_FILE) == @__FILE__
    MetaballsOxygen.start()
end
