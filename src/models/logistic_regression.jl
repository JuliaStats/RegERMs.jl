abstract LogReg <: RegERM

methodname(::LogReg) = "Logistic Regression"
regularizer(::LogReg, w::Vector, λ::Float64) = L2reg(w, λ)

## binomial

immutable BinomialLogReg <: LogReg
    X::Matrix               # n x m matrix of n m-dimensional training examples
    y::Vector               # 1 x n vector with training classes
    n::Int                  # number of training examples
    m::Int                  # number of features
    kernel::Symbol          # kernel function
    regression_type::Symbol # ordinal, binomial, multinomial
end
function BinomialLogReg(X::Matrix, y::Vector; kernel::Symbol=:linear)
    check_arguments(X, y, :binomial)
    BinomialLogReg(X, y, size(X)..., kernel, :binomial)
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
end
function MultinomialLogReg(X::Matrix, y::Vector; kernel::Symbol=:linear)
    check_arguments(X, y, :multinomial)
    MultinomialLogReg(X, y, size(X)..., kernel, :multinomial)
end

loss(::MultinomialLogReg) = MultinomialLogisticLoss()
