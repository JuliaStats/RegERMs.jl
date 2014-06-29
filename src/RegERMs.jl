module RegERMs
using Optim

export RegERM, optimize, objective

abstract RegERM


# FIX: could not use ``method`` as keyword directly due to ``invoke` in linreg, see https://github.com/JuliaLang/julia/issues/7045
optimize(method::RegERM, λ::Float64; optimizer::Symbol=:l_bfgs) = optimize(method, λ, optimizer)
function optimize(method::RegERM, λ::Float64, optimizer::Symbol=:l_bfgs)
	if (λ <= 0)
		throw(ArgumentError("Regularization parameter has to be positive"))
	end

	# init model
	model, X = Model(method.X, method.y, method.kernel)
	y = method.y

	reg(w::Vector) = regularizer(method, w, λ)
	if optimizer == :sgd
		l(w::Vector, i::Int) = loss(method, w, X[i,:], [y[i]])
		grad(w::Vector, i::Int) = vec(gradient(l(w, i))) + gradient(reg(w)) / method.n

		model.w = sgd(grad, method.n, model.w)
	elseif optimizer == :l_bfgs
		l(w::Vector) = loss(method, w, X, y)
		obj(w::Vector) = sum(value(l(w))) + value(reg(w))
		grad(w::Vector) = sum(gradient(l(w)), 2) + gradient(reg(w))
		grad!(w::Vector, storage::Vector) = storage[:] = vec(grad(w))

		res = Optim.optimize(obj, grad!, model.w, method=optimizer, linesearch! = Optim.interpolating_linesearch!)
		model.w = res.minimum
	else
		throw(ArgumentError("Unknown optimizer=$(optimizer)"))
	end
	model
end

function check_arguments(X::Matrix, y::Vector) 
	(n, m) = size(X)
	if (n != length(y))
		throw(DimensionMismatch("Dimensions of X and y mismatch."))
	end
	if (sort(unique(y)) != [-1,1])
		throw(ArgumentError("Class labels have to be either -1 or 1"))
	end
end

# Pretty-print
function Base.show(io::IO, model::RegERM)
	println(io, "$(methodname(model))")
	println(io, repeat("-", length(methodname(model))))
	println(io, "number of examples:       $(model.n)")
	println(io, "number of features:       $(model.m)")
	println(io, "kernel function:          $(model.kernel)")
end

# include 

include("loss.jl")
include("regularizer.jl")
include("mercer_map.jl")
include("model.jl")
# classification methods
include("svm.jl")
include("logistic_regression.jl")
# regression methods
include("ridge_regression.jl")
# sgd
include("sgd.jl")

end # module