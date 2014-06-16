push!(LOAD_PATH, "src")

include("loss.jl")

include("svm.jl")
include("logreg.jl")
include("linreg.jl")