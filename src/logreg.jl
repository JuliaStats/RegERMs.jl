export LogReg

immutable LogReg <: RegERM
    X::Matrix  # n x m matrix of n m-dimensional training examples
    y::Vector  # 1 x n vector with training classes
    λ::Float64 # regularization parameter
    n::Int     # number of training examples
    m::Int     # number of features
end

function LogReg(X::Matrix, y::Vector, λ::Float64)
	(n, m) = size(X)
	if (n != length(y))
		throw(DimensionMismatch("Dimensions of X and y mismatch."))
	end
	if (sort(unique(y)) != [-1,1])
		throw(ArgumentError("Class labels have to be either -1 or 1"))
	end
	LogReg(X, y, λ, n, m)
end

modelname(::LogReg) = "Logistic Regression"
losses{T<:Real}(logreg::LogReg, w::Vector{T}) = logistic(logreg, w)
regularizer{T<:Real}(logreg::LogReg, w::Vector{T}) = l2reg(w, logreg.λ)