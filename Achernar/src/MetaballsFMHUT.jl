module MetaballsFMHUT

import Fomalhaut as FMHUT

using ..MetaballsAX

export start_server

const _WS_PATH = "/metaballs"

function start_server()
    MetaballsAX.init!()

    app = FMHUT.App()
    @FMHUT.websocket app _WS_PATH (ctx) -> nothing

    @async FMHUT.serve(app; fps=30)
    sleep(0.5)

    try
        start_time = time()
        interval   = 1.0 / 30.0

        while FMHUT._server_running[]
            frame_start = time()
            elapsed_total = frame_start - start_time

            MetaballsAX.update_physics!(elapsed_total)
            MetaballsAX.compute_field!(MetaballsAX.FIELD_BUFFER)
            vertex_float_count = MetaballsAX.build_mesh!(
                MetaballsAX.VERTEX_BUFFER,
                MetaballsAX.NORMAL_BUFFER,
                MetaballsAX.FIELD_BUFFER
            )
            payload = MetaballsAX.build_payload!(
                MetaballsAX.VERTEX_BUFFER,
                MetaballsAX.NORMAL_BUFFER,
                vertex_float_count
            )

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
            @info "Metaballs server interrupted"
        else
            rethrow(err)
        end
    end
end

end # module MetaballsFMHUT

if abspath(PROGRAM_FILE) == @__FILE__
    MetaballsFMHUT.start_server()
end
