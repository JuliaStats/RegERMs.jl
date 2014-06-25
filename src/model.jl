export Model, PrimalModel, DualModel, classify

function Model(X::Matrix, y::Vector)
	n, m = size(X)
	if m > n # learn dual model to reduce number of dimensions
		Model(X, y, :linear)
	else
		w0 = vec(mean(X[y.==1,:],1).-mean(X[y.==-1,:]))
		PrimalModel(w0), X
	end
end

function Model(X::Matrix, y::Vector, kernel::Symbol)
	n, m = size(X)
	if m < n && kernel==:linear # learn primal model to reduce number of dimensions
		Model(X, y)
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

## Dual model

# TOOD(cs): map should be immutable
type DualModel
	w::Vector 	   # weight vector of linear function
	map::MercerMap # dual model is implemented via Mercer map
end

classify(model::DualModel, X::Matrix)  = sign(apply(model.map, X)*model.w)