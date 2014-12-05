abstract LogReg <: RegERM

methodname(::LogReg) = "Logistic Regression"
regularizer(LogReg::LogReg, w::Vector) = L2reg(w, LogReg.params.λ)

## binomial

immutable BinomialLogReg <: LogReg
    X::Matrix               # n x m matrix of n m-dimensional training examples
    y::Vector               # 1 x n vector with training classes
    n::Int                  # number of training examples
    m::Int                  # number of features
    kernel::Symbol          # kernel function
    regression_type::Symbol # ordinal, binomial, multinomial
    params::Hyperparameters # hyperparameters (e.g. λ)
end
function BinomialLogReg(X::Matrix, y::Vector; kernel::Symbol=:linear, λ::Float64=0.1)
    check_arguments(X, y, :binomial)
    BinomialLogReg(X, y, size(X)..., kernel, :binomial, RegularizationParameters(λ))
end

loss(::BinomialLogReg) = LogisticLoss()

## multinomial

immutable MultinomialLogReg <: LogReg
    X::Matrix               # n x m matrix of n m-dimensional training examples
    y::Vector               # 1 x n vector with training classes
    n::Int                  # number of training examples
    m::Int                  # number of features
    kernel::Symbol          # kernel function
    regression_type::Symbol # ordinal, binomial, multinomial
    params::Hyperparameters # hyperparameters (e.g. λ)
end
function MultinomialLogReg(X::Matrix, y::Vector; kernel::Symbol=:linear, λ::Float64=0.1)
    check_arguments(X, y, :multinomial)
    MultinomialLogReg(X, y, size(X)..., kernel, :multinomial, RegularizationParameters(λ))
end

loss(::MultinomialLogReg) = MultinomialLogisticLoss()
