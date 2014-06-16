module RegERMs
using Optim

export RegERM, optimize, objective

abstract RegERM

# Pretty-print
function Base.show(io::IO, model::RegERM)
	println(io, "$(modelname(model))")
	println(io, repeat("-", length(modelname(model))))
	println(io, "regularization parameter: $(model.λ)")
	println(io, "number of examples:       $(model.n)")
	println(io, "number of features:       $(model.m)")
end

# cannot use method as keyword due to invoke in linreg: https://github.com/JuliaLang/julia/issues/7045
optimize(model::RegERM, method=:l_bfgs) = optimize(model, zeros(Float64, model.m), method)
function optimize(model::RegERM, w0::Vector, method=:l_bfgs)
	X = model.X
	y = model.y

	if method == :sgd
		grad(w::Vector, i::Int) = vec(gradient(losses(model, w, i)) + gradient(regularizer(model, w)) / model.n)
		sgd(grad, model.n, w0)
	elseif method ==:l_bfgs
		obj(w::Vector) = objective(model, w)		
		grad(w::Vector) = sum(gradient(losses(model, w)), 2) + gradient(regularizer(model, w))
		grad!(w::Vector, storage::Vector) = storage[:] = vec(grad(w))

		Optim.optimize(obj, grad!, w0, method=:l_bfgs, linesearch! = Optim.interpolating_linesearch!).minimum
	else
		throw(ArgumentError("Unknown optimization method=$(method)"))
	end
end

objective(model::RegERM, w::Vector) = sum(value(losses(model, w))) + value(regularizer(model, w))


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