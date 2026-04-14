module PhillipsOceanFMHUT

using Sockets
include("PhillipsOcean.jl")
using .PhillipsOcean

# Cross-platform IPC path
const SOCKET_PATH = Sys.iswindows() ? 
    raw"\\.\pipe\phillips_ocean" : 
    "/tmp/phillips_ocean.sock"

function start_server()
    # Clean up old sockets ( Windows named pipes do not need rm )
    if !Sys.iswindows()
        rm(SOCKET_PATH, force=true)
    end

    println("Starting Julia IPC Server at: $SOCKET_PATH")
    server = listen(SOCKET_PATH)
    @info "Julia IPC Server started successfully ($(Sys.iswindows() ? "Windows Named Pipe" : "Unix Domain Socket"))"

    try
        while true
            conn = accept(server)
            @info "Rust client connected."

            @async begin
                try
                    start_time = time()
                    while isopen(conn)
                        frame_start = time()
                        t = frame_start - start_time

                        compute_wave!(FRAME_BUFFER, t)

                        # Send raw binary data to Rust
                        write(conn, reinterpret(UInt8, FRAME_BUFFER))

                        # Control the frame rate ( ~30 FPS )
                        elapsed = time() - frame_start
                        sleep(max(0.0, FRAME_INTERVAL - elapsed))
                    end
                catch e
                    @warn "Connection error: $e"
                finally
                    close(conn)
                end
            end
        end
    catch e
        @error "Server error: $e"
    finally
        close(server)
        if !Sys.iswindows()
            rm(SOCKET_PATH, force=true)
        end
        @info "Julia IPC Server shutdown."
    end
end

end  # module

# Start the server by executing this file directly
if abspath(PROGRAM_FILE) == @__FILE__
    PhillipsOceanFMHUT.start_server()
end