module RegERMs
using Optim

export RegErm, train

abstract RegErm

function train(model::RegErm, lambda::Real)
	w0 = [1.0*x for x in zeros(model.num_features,1)]

	obj(w::Vector) = sum(losses(model, w)[1]) + lambda*regularizer(model, w)[1]
	grad(w::Vector) = sum(losses(model, w)[2]) + lambda*regularizer(model, w)[2]
	optimize(obj, w0, method=:l_bfgs).minimum
end

include("svm.jl")

end # module