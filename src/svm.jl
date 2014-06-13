export SVM

immutable SVM <: RegErm
    X::Matrix
    y::Vector
    num_features::Int
    num_examples::Int
end

function SVM(X::Matrix, y::Vector)
	(n, m) = size(X)
	if (n != length(y))
		error("dimension mismatch. Try: X'")
	end
	SVM(X, y, m, n)
end

function hinge{T<:Real}(svm::SVM, w::Vector{T})
	f = svm.X*w
	losses = max(0, 1.-svm.y.*f)
    gradient = -svm.y.*(losses .> 0)

    (losses, gradient)
end

l2norm{T<:Real}(w::Vector{T}) = (norm(w)^2/2, w)

losses{T<:Real}(svm::SVM, w::Vector{T}) = hinge(svm, w)
regularizer{T<:Real}(svm::SVM, w::Vector{T}) = l2norm(w)

# Pretty-print
function Base.show(io::IO, svm::SVM)
	println(io, "SVM:")
	println(io, "number of examples: $(svm.num_examples)")
	println(io, "number of features: $(svm.num_features)")
end