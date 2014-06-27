export LinReg

immutable LinReg <: RegERM
    X::Matrix  # n x m matrix of n m-dimensional training examples
    y::Vector  # 1 x n vector with training classes
    n::Int     # number of training examples
    m::Int     # number of features
end

function LinReg(X::Matrix, y::Vector)
	check_arguments(X, y)
	LinReg(X, y, size(X)...)
end

modelname(::LinReg) = "Linear Regression"
loss(::LinReg, w::Vector, X::Matrix, y::Vector) = Squared(w, X, y)
regularizer(::LinReg, w::Vector, λ::Float64) = L2reg(w, λ)

# closed-form solution
function optimize(linreg::LinReg, w0::Vector, λ::Float64; method::Symbol=:closed_form)
	if method == :closed_form
		X = linreg.X
		y = linreg.y
		(X'*X + eye(linreg.m)/λ)\X'*y
	else
		invoke(optimize, (RegERM, Vector, Float64, Symbol), linreg, w0, λ, method)
	end
end