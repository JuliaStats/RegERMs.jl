export LinReg

immutable LinReg <: RegERM
    X::Matrix       # n x m matrix of n m-dimensional training examples
    y::Vector       # 1 x n vector with training classes
    n::Int          # number of training examples
    m::Int          # number of features
    kernel::Symbol 	# kernel function
end

function LinReg(X::Matrix, y::Vector; kernel::Symbol=:linear)
	check_arguments(X, y)
	LinReg(X, y, size(X)..., kernel)
end

methodname(::LinReg) = "Linear Regression"
loss(::LinReg, w::Vector, X::Matrix, y::Vector) = Squared(w, X, y)
regularizer(::LinReg, w::Vector, λ::Float64) = L2reg(w, λ)

# closed-form solution
function optimize(linreg::LinReg, λ::Float64; optimizer::Symbol=:closed_form)
	if (λ <= 0)
		throw(ArgumentError("Regularization parameter has to be positive"))
	end
	
	if optimizer == :closed_form
		y = linreg.y
		model, X = Model(linreg.X, y, linreg.kernel)
		model.w = (X'*X + eye(linreg.m)/λ)\X'*y
		model
	else
		invoke(optimize, (RegERM, Float64, Symbol), linreg, λ, optimizer)
	end
end