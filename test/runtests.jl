push!(LOAD_PATH, "src")

using RegERMs, Base.Test

include("loss.jl")
include("model.jl")
include("mercer_map.jl")

include("svm.jl")
include("logistic_regression.jl")
include("ridge_regression.jl")