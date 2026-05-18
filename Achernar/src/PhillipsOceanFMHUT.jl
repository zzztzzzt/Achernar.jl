module PhillipsOceanFMHUT

using Libdl
import Fomalhaut as FMHUT
import Axis as AX

using ..PhillipsOceanAX

export start_server

"""
    start_server()

Start the direct Rust-to-Rust WGPU bridge server.
No heap allocation in the hot path — Julia only drives the timing loop.
"""
function start_server()
    # 1. Initialize the ocean simulation ( wgpu device + GPU buffers + shader )
    PhillipsOceanAX.init!()

    # 2. Setup Fomalhaut app and register the WebSocket route
    app = FMHUT.App()
    @FMHUT.websocket app "/phillips-ocean" (ctx) -> nothing

    # 3. Wire the Rust-to-Rust broadcast bridge:
    #    Obtain Fomalhaut's fmh_ws_broadcast C function pointer and register it
    #    in axis_rs so compute_wave_and_broadcast! can call it directly —
    #    no Julia involvement in the data path whatsoever.
    fmhut_handle  = Libdl.dlopen(FMHUT._load_rust_lib())
    broadcast_ptr = Libdl.dlsym(fmhut_handle, :fmh_ws_broadcast)
    AX.axis_set_broadcast_callback(broadcast_ptr)

    @info "Rust-to-Rust Direct Bridge established! Server starting..."

    # 4. Start Fomalhaut server asynchronously
    @async FMHUT.serve(app; fps=60)
    sleep(0.5)  # give the server time to bind

    # 5. Main simulation loop
    #    Julia only controls timing; all data stays inside Rust until the WebSocket.
    try
        start_time = time()
        interval   = 1.0 / 60.0

        while FMHUT._server_running[]
            frame_start = time()
            t = frame_start - start_time

            # Zero Julia allocation:
            #   pack (Rust) → GPU dispatch → Rust readback → Envelope V1 → broadcast
            PhillipsOceanAX.compute_wave_and_broadcast!(t, "/phillips-ocean")

            elapsed      = time() - frame_start
            target_sleep = interval - elapsed

            # Precision sleep: coarse sleep + spin for sub-millisecond accuracy on Windows
            if target_sleep > 0.002
                sleep(target_sleep - 0.0015)
            end
            while time() - frame_start < interval
                yield()  # spin for remaining sub-millisecond precision
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
