module PhillipsOceanFMHUT

using Fomalhaut

include("PhillipsOcean.jl")
using .PhillipsOcean

function wave_stream(ctx)
    compute_wave!(FRAME_BUFFER, ctx.time)
    return FRAME_BUFFER 
end

# Start websocket backend and stream computed frames via Fomalhaut FFI
function start_server()
    app = App()

    @websocket app "/phillips-ocean" wave_stream

    Fomalhaut.serve(app; fps=60)
end

end # module PhillipsOceanFMHUT

# Start the stream server when this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    PhillipsOceanFMHUT.start_server()
end
