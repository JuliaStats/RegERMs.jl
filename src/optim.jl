
abstract RegERM
abstract RegressionSolver

# FIX: could not use ``method`` as keyword directly due to ``invoke` in RidgeReg, see https://github.com/JuliaLang/julia/issues/7045
optimize(method::RegERM, λ::Float64; optimizer::Symbol=:l_bfgs) = optimize(method, λ, optimizer)
function optimize(method::RegERM, λ::Float64, optimizer::Symbol=:l_bfgs)
    if (λ <= 0)
        throw(ArgumentError("Regularization parameter has to be positive"))
    end

    # init model
    model = Model(method.X, method.y, method.regression_type, method.kernel)

    if optimizer == :sgd
        model.theta = solve(model, method, SGDSolver(), method.X, method.y, λ)
    elseif optimizer == :l_bfgs
        model.theta = solve(model, method, LBFGSSolver(), method.X, method.y, λ)
    else
        throw(ArgumentError("Unknown optimizer=$(optimizer)"))
    end
    model
end

objective(method::RegERM, model::RegressionModel, λ::Float64, theta::AbstractVector) =
    tloss(loss(method), values(model, method.X, theta), method.y) + value(regularizer(method, theta, λ))

function check_arguments(X::Matrix, y::Vector) 
    (n, m) = size(X)
    if (n != length(y))
        throw(DimensionMismatch("Dimensions of X and y mismatch."))
    end
    if (sort(unique(y)) != [-1,1])
        throw(ArgumentError("Class labels have to be either -1 or 1"))
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
