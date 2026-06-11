module Achernar

export PhillipsOceanAX, MetaballsAX

include("PhillipsOcean.jl")
include("PhillipsOceanOxygen.jl")
include("PhillipsOceanAX.jl")
include("PhillipsOceanFMHUT.jl")

using .PhillipsOcean
using .PhillipsOceanOxygen
using .PhillipsOceanAX
using .PhillipsOceanFMHUT

include("Metaballs.jl")
include("MetaballsOxygen.jl")
include("MetaballsAX.jl")
include("MetaballsFMHUT.jl")

using .Metaballs
using .MetaballsOxygen
using .MetaballsAX
using .MetaballsFMHUT

end # module Achernar
