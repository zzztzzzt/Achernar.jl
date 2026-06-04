module Achernar

include("PhillipsOcean.jl")
using .PhillipsOcean

include("PhillipsOceanOxygen.jl")
using .PhillipsOceanOxygen

include("PhillipsOceanAX.jl")
using .PhillipsOceanAX
export PhillipsOceanAX

include("PhillipsOceanFMHUT.jl")
using .PhillipsOceanFMHUT

include("Metaballs.jl")
using .Metaballs

include("MetaballsOxygen.jl")
using .MetaballsOxygen

end # module Achernar
