immutable RidgeReg <: RegERM
    X::Matrix               # n x m matrix of n m-dimensional training examples
    y::Vector               # 1 x n vector with training classes
    n::Int                  # number of training examples
    m::Int                  # number of features
    kernel::Symbol          # kernel function
    regression_type::Symbol # ordinal, binomial, multinomial
    params::Hyperparameters # hyperparameters (e.g. λ)
end

function RidgeReg(X::Matrix, y::Vector; kernel::Symbol=:linear, λ::Float64=0.1)
    check_arguments(X, y, :ordinal)
    RidgeReg(X, y, size(X)..., kernel, :ordinal, RegularizationParameters(λ))
end

methodname(::RidgeReg) = "Linear Regression"
loss(::RidgeReg) = SquaredLoss()
regularizer(RidgeReg::RidgeReg, w::Vector) = L2reg(w, RidgeReg.params.λ)

# closed-form solution
function optimize(RidgeReg::RidgeReg; optimizer::Symbol=:closed_form)
    check_hyperparameters(RidgeReg.params)
    
    if optimizer == :closed_form
        y, X = RidgeReg.y, RidgeReg.X
        model = Model(X, y, RidgeReg.regression_type, RidgeReg.kernel)
        model.theta = (X'*X + eye(RidgeReg.m)/RidgeReg.params.λ)\X'*y
        model
    else
        invoke(optimize, (RegERM, Symbol), RidgeReg, optimizer)
    end
end