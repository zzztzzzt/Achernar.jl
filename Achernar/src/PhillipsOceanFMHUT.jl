module PhillipsOceanFMHUT

using Fomalhaut

include("PhillipsOcean.jl")
using .PhillipsOcean

# Start websocket backend and stream computed frames via Fomalhaut FFI
function start_server(; host::AbstractString = "127.0.0.1", port::Integer = 8080)
    Fomalhaut.start_server(host = host, port = port)
    @info "Fomalhaut WS backend started at ws://$(host):$(port)"

    start_time = time()
    try
        while true
            frame_start = time()
            t = frame_start - start_time

            # Compute frame data in Julia
            compute_wave!(FRAME_BUFFER, t)

            # Convert Float32 frame buffer to raw bytes and send
            payload = reinterpret(UInt8, FRAME_BUFFER) |> collect
            Fomalhaut.send_frame!(
                payload;
                content_type = Fomalhaut.CONTENT_TYPE_FLOAT32_TENSOR,
            )

            # Keep target frame rate ( for example ~30 FPS )
            elapsed = time() - frame_start
            sleep(max(0.0, FRAME_INTERVAL - elapsed))
        end
    catch e
        @error "Streaming loop error: $e"
        rethrow()
    finally
        # Ensure backend stops on interruption / error
        try
            Fomalhaut.stop_server!()
            @info "Fomalhaut WS backend stopped."
        catch stop_err
            @warn "Failed to stop backend cleanly: $stop_err"
        end
    end
end

end # module PhillipsOceanFMHUT

# Start the stream server when this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    PhillipsOceanFMHUT.start_server()
end
