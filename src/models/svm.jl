immutable SVM <: RegERM
    X::Matrix               # n x m matrix of n m-dimensional training examples
    y::Vector               # 1 x n vector with training classes
    n::Int                  # number of training examples
    m::Int                  # number of features
    kernel::Symbol          # kernel function
    regression_type::Symbol # ordinal, binomial, multinomial
end

function SVM(X::Matrix, y::Vector; kernel::Symbol=:linear)
    check_arguments(X, y)
    SVM(X, y, size(X)..., kernel, :binomial)
end

methodname(::SVM) = "Support Vector Machine"
loss(::SVM) = HingeLoss()
regularizer(::SVM, w::Vector, λ::Float64) = L2reg(w, λ)
