
abstract RegERM
abstract RegressionSolver

# FIX: could not use ``method`` as keyword directly due to ``invoke` in RidgeReg, see https://github.com/JuliaLang/julia/issues/7045
optimize(method::RegERM, optimizer::Symbol=:l_bfgs) = optimize(method, optimizer)

function optimize(method::RegERM, optimizer::Symbol=:l_bfgs)
    check_hyperparameters(method.params)

    # init model
    model = Model(method.X, method.y, method.regression_type, method.kernel)

    if optimizer == :sgd
        model.theta = solve(model, method, SGDSolver(), method.X, method.y)
    elseif optimizer == :l1_rda && typeof(method.params) == L1RDAParameters
        model.theta = solve(model, method, L1RDASolver(), method.X, method.y)
    elseif optimizer == :l_bfgs
        model.theta = solve(model, method, LBFGSSolver(), method.X, method.y)
    else
        throw(ArgumentError("Unknown optimizer=$(optimizer) or mismatched hyperparameters"))
    end
    model
end

function check_arguments(X::Matrix, y::Vector, regression_type::Symbol)
    (n, m) = size(X)
    if (n != length(y))
        throw(DimensionMismatch("Dimensions of X and y mismatch."))
    end
    if (regression_type == :binomial)
        if (sort(unique(y)) != [-1,1])
            throw(ArgumentError("Class labels have to be either -1 or 1"))
        end
    elseif (regression_type == :multinomial)
        if (typeof(y) != Array{Int,1} || any(y.<1))
            throw(ArgumentError("Classes have to be positive integer values"))
        end
    elseif (regression_type == :ordinal)
        # something additional to check for regression?
    else
        throw(ArgumentError("Unknown regression_type=$(regression_type)"))
    end
end

# Pretty-print
function Base.show(io::IO, model::RegERM)
    println(io, "$(methodname(model))")
    println(io, repeat("-", length(methodname(model))))
    println(io, "number of examples:       $(model.n)")
    println(io, "number of features:       $(model.m)")
    println(io, "kernel function:          $(model.kernel)")
end
