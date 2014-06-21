module RegERMs
using Optim

export RegERM, optimize, objective

abstract RegERM

loss(model::RegERM, w::Vector) = loss(model, w, model.X, model.y)
loss(model::RegERM, w::Vector, i::Int) = loss(model, w, model.X[i,:], [model.y[i]])
regularizer(model::RegERM, w::Vector) = regularizer(model, w, model.λ)
objective(model::RegERM, w::Vector) = sum(value(loss(model, w))) + value(regularizer(model, w))

# FIX: could not use ``method`` as keyword directly due to ``invoke` in linreg, see https://github.com/JuliaLang/julia/issues/7045
optimize(model::RegERM, w0::Vector; method::Symbol=:l_bfgs) = optimize(model, w0, method)
function optimize(model::RegERM, w0::Vector, method::Symbol=:l_bfgs)
	if method == :sgd
		grad(w::Vector, i::Int) = vec(gradient(loss(model, w, i)) + gradient(regularizer(model, w)) / model.n)
		
		sgd(grad, model.n, w0)
	elseif method == :l_bfgs
		obj(w::Vector) = objective(model, w)		
		grad(w::Vector) = sum(gradient(loss(model, w)), 2) + gradient(regularizer(model, w))
		grad!(w::Vector, storage::Vector) = storage[:] = vec(grad(w))

		Optim.optimize(obj, grad!, w0, method=:l_bfgs, linesearch! = Optim.interpolating_linesearch!).minimum
	else
		throw(ArgumentError("Unknown optimization method=$(method)"))
	end
end
optimize(model::RegERM; method=:l_bfgs) = optimize(model, zeros(Float64, model.m); method=method)

function check_arguments(X::Matrix, y::Vector, λ::Float64) 
	(n, m) = size(X)
	if (n != length(y))
		throw(DimensionMismatch("Dimensions of X and y mismatch."))
	end
	if (sort(unique(y)) != [-1,1])
		throw(ArgumentError("Class labels have to be either -1 or 1"))
	end
	if (λ <= 0)
		throw(ArgumentError("Regularization parameter has to be positive"))
	end
end

# Pretty-print
function Base.show(io::IO, model::RegERM)
	println(io, "$(modelname(model))")
	println(io, repeat("-", length(modelname(model))))
	println(io, "regularization parameter: $(model.λ)")
	println(io, "number of examples:       $(model.n)")
	println(io, "number of features:       $(model.m)")
end

# include 

include("loss.jl")
include("regularizer.jl")
# classification models
include("svm.jl")
include("logreg.jl")
# regression models
include("linreg.jl")
# sgd
include("sgd.jl")

end # module