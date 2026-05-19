module PhillipsOceanFMHUT

import Fomalhaut as FMHUT

using ..PhillipsOceanAX

export start_server

const _WS_PATH = "/phillips-ocean"

function start_server()
    PhillipsOceanAX.init!()

    app = FMHUT.App()
    @FMHUT.websocket app _WS_PATH (ctx) -> nothing

    @async FMHUT.serve(app; fps=60)
    sleep(0.5)

    try
        start_time = time()
        interval   = 1.0 / 60.0

        while FMHUT._server_running[]
            frame_start = time()
            t = frame_start - start_time

            PhillipsOceanAX.compute_wave!(PhillipsOceanAX.FRAME_BUFFER, t)
            payload = vec(reinterpret(UInt8, PhillipsOceanAX.FRAME_BUFFER))
            FMHUT.broadcast_frame!(_WS_PATH, payload)

            elapsed      = time() - frame_start
            target_sleep = interval - elapsed

            if target_sleep > 0.002
                sleep(target_sleep - 0.0015)
            end
            while time() - frame_start < interval
                yield()
            end
        end
    catch err
        if err isa InterruptException
            @info "Phillips ocean server interrupted"
        else
            rethrow(err)
        end
    end
end

end # module PhillipsOceanFMHUT

if abspath(PROGRAM_FILE) == @__FILE__
    PhillipsOceanFMHUT.start_server()
end
