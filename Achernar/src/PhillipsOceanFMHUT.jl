module PhillipsOceanFMHUT

using Libdl
import Fomalhaut as FMHUT
import Axis.Ocean as AXOcean

using ..PhillipsOceanAX

export start_server

"""
    start_server()

Start the direct Rust-to-Rust WGPU bridge server.
No heap allocation in the hot path.
"""
function start_server()
    # 1. Initialize Axis ocean simulation
    PhillipsOceanAX.init!()
    
    # 2. Setup Fomalhaut app
    app = FMHUT.App()
    
    # Register websocket route
    @FMHUT.websocket app "/phillips-ocean" (ctx) -> nothing
    
    # 3. Get Fomalhaut's fmh_ws_broadcast pointer
    fmhut_lib_path = FMHUT._load_rust_lib()
    fmhut_handle = Libdl.dlopen(fmhut_lib_path)
    broadcast_ptr = Libdl.dlsym(fmhut_handle, :fmh_ws_broadcast)
    
    # 4. Set the callback in Axis Rust side
    AXOcean.axis_set_broadcast_callback(broadcast_ptr)
    
    @info "Rust-to-Rust Direct Bridge established! Server starting..."
    
    # 5. Start Fomalhaut server asynchronously
    @async FMHUT.serve(app; fps=60)
    
    # Give server time to bind
    sleep(0.5)
    
    # 6. Run the native simulation loop in Julia
    try
        start_time = time()
        interval = 1.0 / 120.0
        while FMHUT._server_running[]
            frame_start = time()
            t = frame_start - start_time
            
            # Dispatch compute, read directly in Rust, package Envelope, and call broadcast
            AXOcean.compute_wave_and_broadcast!(PhillipsOceanAX._SIM[], t, "/phillips-ocean")
            
            elapsed = time() - frame_start
            target_sleep = interval - elapsed
            
            # Precision sleep for perfectly smooth 120 FPS on Windows
            if target_sleep > 0.002
                sleep(target_sleep - 0.0015)
            end
            while time() - frame_start < interval
                yield() # Spin for the remaining sub-millisecond precision
            end
        end
    catch err
        if err isa InterruptException
            @info "Direct server interrupted"
        else
            rethrow(err)
        end
    end
end

end # module PhillipsOceanFMHUT

# Start the stream server when this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    PhillipsOceanFMHUT.start_server()
end
