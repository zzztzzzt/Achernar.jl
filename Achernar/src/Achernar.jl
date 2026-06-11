module Achernar

export PhillipsOceanAX, MetaballsAX

include("PhillipsOcean.jl")
include("PhillipsOceanOxygen.jl")
include("PhillipsOceanAX.jl")
include("PhillipsOceanFMHUT.jl")

include("Metaballs.jl")
include("MetaballsOxygen.jl")
include("MetaballsAX.jl")
include("MetaballsFMHUT.jl")

using .PhillipsOcean
using .PhillipsOceanOxygen
using .PhillipsOceanAX
using .PhillipsOceanFMHUT

using .Metaballs
using .MetaballsOxygen
using .MetaballsAX
using .MetaballsFMHUT

end # module Achernar
