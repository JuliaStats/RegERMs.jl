export LogReg

immutable LogReg <: RegERM
    X::Matrix           # n x m matrix of n m-dimensional training examples
    y::Vector           # 1 x n vector with training classes
    λ::Float64          # regularization parameter
    num_features::Int   # number of features
    num_examples::Int   # number of training examples
end

function LogReg(X::Matrix, y::Vector, λ::Float64)
	(n, m) = size(X)
	if (n != length(y))
		error("dimension mismatch. Try: X'")
	end
	LogReg(X, y, λ, m, n)
end

modelname(logreg::LogReg) = "Logistic Regression"
losses{T<:Real}(logreg::LogReg, w::Vector{T}) = logistic(logreg, w)
regularizer{T<:Real}(logreg::LogReg, w::Vector{T}) = l2reg(w, logreg.λ)