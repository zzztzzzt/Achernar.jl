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

end # module Achernar
