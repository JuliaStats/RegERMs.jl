immutable RidgeReg <: RegERM
    X::Matrix       # n x m matrix of n m-dimensional training examples
    y::Vector       # 1 x n vector with training classes
    n::Int          # number of training examples
    m::Int          # number of features
    kernel::Symbol  # kernel function
end

function RidgeReg(X::Matrix, y::Vector; kernel::Symbol=:linear)
    check_arguments(X, y)
    RidgeReg(X, y, size(X)..., kernel)
end

methodname(::RidgeReg) = "Linear Regression"
loss(::RidgeReg) = SquaredLoss()
regularizer(::RidgeReg, w::Vector, λ::Float64) = L2reg(w, λ)

# closed-form solution
function optimize(RidgeReg::RidgeReg, λ::Float64; optimizer::Symbol=:closed_form)
    if (λ <= 0)
        throw(ArgumentError("Regularization parameter has to be positive"))
    end
    
    if optimizer == :closed_form
        y = RidgeReg.y
        model, X = Model(RidgeReg.X, y, RidgeReg.kernel)
        model.w = (X'*X + eye(RidgeReg.m)/λ)\X'*y
        model
    else
        invoke(optimize, (RegERM, Float64, Symbol), RidgeReg, λ, optimizer)
    end
end