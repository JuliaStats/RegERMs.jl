export LogReg

immutable LogReg <: RegERM
    X::Matrix       # n x m matrix of n m-dimensional training examples
    y::Vector       # 1 x n vector with training classes
    n::Int          # number of training examples
    m::Int          # number of features
    kernel::Symbol 	# kernel function 
end

function LogReg(X::Matrix, y::Vector; kernel::Symbol=:linear)
	check_arguments(X, y)
	LogReg(X, y, size(X)..., kernel)
end

methodname(::LogReg) = "Logistic Regression"
loss(::LogReg) = LogisticLoss()
regularizer(::LogReg, w::Vector, λ::Float64) = L2reg(w, λ)