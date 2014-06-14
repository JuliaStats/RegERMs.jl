export LinReg

immutable LinReg <: RegERM
    X::Matrix  # n x m matrix of n m-dimensional training examples
    y::Vector  # 1 x n vector with training classes
    λ::Float64 # regularization parameter
    n::Int     # number of training examples
    m::Int     # number of features
end

function LinReg(X::Matrix, y::Vector, λ::Float64)
	(n, m) = size(X)
	if (n != length(y))
		throw(DimensionMismatch("Dimensions of X and y mismatch."))
	end
	LinReg(X, y, λ, n, m)
end

modelname(::LinReg) = "Linear Regression"
losses(linreg::LinReg, w::Vector) = Squared(w, linreg.X, linreg.y)
losses(linreg::LinReg, w::Vector, i::Int) = Squared(w, linreg.X[i,:], [linreg.y[i]])
regularizer(linreg::LinReg, w::Vector) = L2reg(w, linreg.λ)

# closed-form solution
function optimize(linreg::LinReg, method=:closed_form)
	if method == :closed_form
		X = linreg.X
		y = linreg.y
		(X'*X + eye(linreg.m)/linreg.λ)\X'*y
	else
		invoke(optimize, (RegERM, Symbol), linreg, method)
	end
end