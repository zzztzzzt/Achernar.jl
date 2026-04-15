module PhillipsOceanFMHUT

using Fomalhaut

include("PhillipsOcean.jl")
using .PhillipsOcean

# Start websocket backend and stream computed frames via Fomalhaut FFI
function start_server()
    @stream "/wave" (ctx) -> begin
        compute_wave!(FRAME_BUFFER, ctx.time)
        FRAME_BUFFER
    end

    Fomalhaut.start()
end

end # module PhillipsOceanFMHUT

# Start the stream server when this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    PhillipsOceanFMHUT.start_server()
end
