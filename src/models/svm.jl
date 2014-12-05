immutable SVM <: RegERM
    X::Matrix               # n x m matrix of n m-dimensional training examples
    y::Vector               # 1 x n vector with training classes
    n::Int                  # number of training examples
    m::Int                  # number of features
    kernel::Symbol          # kernel function
    regression_type::Symbol # ordinal, binomial, multinomial
    params::Hyperparameters # hyperparameters (e.g. λ)
end

function SVM(X::Matrix, y::Vector; kernel::Symbol=:linear, λ::Float64=0.1)
    check_arguments(X, y, :binomial)
    SVM(X, y, size(X)..., kernel, :binomial, RegularizationParameters(λ))
end

function SVM(X::Matrix, y::Vector, params::Hyperparameters; kernel::Symbol=:linear)
    check_hyperparameters(params)
    check_arguments(X, y, :binomial)
    SVM(X, y, size(X)..., kernel, :binomial, params)
end

methodname(::SVM) = "Support Vector Machine"
loss(::SVM) = HingeLoss()
regularizer(SVM::SVM, w::Vector) = L2reg(w, SVM.params.λ)
