export Model, PrimalModel, DualModel, classify

function Model(X::Matrix, y::Vector, kernel::Symbol=:linear)
	n, m = size(X)
	if m < n && kernel==:linear # learn primal model to reduce number of dimensions
		w0 = vec(mean(X[y.==1,:],1).-mean(X[y.==-1,:]))
		PrimalModel(w0), X
	else
		map = MercerMap(X, kernel)
		X = apply(map)
		w0 = vec(mean(X[y.==1,:],1).-mean(X[y.==-1,:]))
		DualModel(w0, map), X
	end
end

## Primal model

type PrimalModel
	w::Vector # weight vector of linear function
end

classify(model::PrimalModel, X::Matrix) = sign(X*model.w)

# Pretty-print
function Base.show(io::IO, model::PrimalModel)
	println(io, "Primal Model")
	println(io, repeat("-", length("Primal Model")))
	println(io, "number of dimensions: $(length(model.w))")
end

## Dual model

# TOOD(cs): map should be immutable
type DualModel
	w::Vector 	   # weight vector of linear function
	map::MercerMap # dual model is implemented via Mercer map
end

classify(model::DualModel, X::Matrix)  = sign(apply(model.map, X)*model.w)

# Pretty-print
function Base.show(io::IO, model::DualModel)
	println(io, "Dual Model")
	println(io, repeat("-", length("Dual Model")))
	println(io, "number of dimensions: $(length(model.w))")
	println(io, "number of examples:   $(size(model.map.K,1))")
	println(io, "kernel function:      $(model.map.kernel)")
end