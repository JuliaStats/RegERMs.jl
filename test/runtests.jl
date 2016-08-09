using RegERMs, Base.Test


tests = [
    "loss",
    "model", 
    "mercer_map",
    "svm", 
    "logistic_regression", 
    "ridge_regression"
]

println("Running tests:")

for t in tests
    test_fn = "$t.jl"
    println("* $test_fn")
    include(test_fn)
end
