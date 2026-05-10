module PhillipsOceanFMHUT

import Fomalhaut as FMHUT

include("PhillipsOceanAX.jl")
using .PhillipsOceanAX

function wave_stream(ctx)
    compute_wave!(FRAME_BUFFER, ctx.time)
    return FRAME_BUFFER 
end

# Start websocket backend and stream computed frames via Fomalhaut FFI
function start_server()
    init!()
    
    app = FMHUT.App()

    @FMHUT.websocket app "/phillips-ocean" wave_stream

    FMHUT.serve(app; fps=120)
end

end # module PhillipsOceanFMHUT

# Start the stream server when this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    PhillipsOceanFMHUT.start_server()
end
