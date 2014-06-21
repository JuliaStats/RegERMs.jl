export SVM

immutable SVM <: RegERM
    X::Matrix  # n x m matrix of n m-dimensional training examples
    y::Vector  # 1 x n vector with training classes
    λ::Float64 # regularization parameter
    n::Int     # number of training examples
    m::Int     # number of features
end

function SVM(X::Matrix, y::Vector, λ::Float64)
	check_arguments(X, y, λ)
	SVM(X, y, λ, size(X)...)
end

modelname(::SVM) = "Support Vector Machine"
loss(::SVM, w::Vector, X::Matrix, y::Vector) = Hinge(w, X, y)
regularizer(::SVM, w::Vector, λ::Float64) = L2reg(w, λ)
