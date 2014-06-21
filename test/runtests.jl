push!(LOAD_PATH, "src")

include("loss.jl")

include("svm.jl")
include("logistic_regression.jl")
include("ridge_regression.jl")