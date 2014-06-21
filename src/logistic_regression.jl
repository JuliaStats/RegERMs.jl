export LogReg

immutable LogReg <: RegERM
    X::Matrix  # n x m matrix of n m-dimensional training examples
    y::Vector  # 1 x n vector with training classes
    λ::Float64 # regularization parameter
    n::Int     # number of training examples
    m::Int     # number of features
end

function LogReg(X::Matrix, y::Vector, λ::Float64)
	check_arguments(X, y, λ)
	LogReg(X, y, λ, size(X)...)
end

modelname(::LogReg) = "Logistic Regression"
loss(::LogReg, w::Vector, X::Matrix, y::Vector) = Logistic(w, X, y)
regularizer(::LogReg, w::Vector, λ::Float64) = L2reg(w, λ)