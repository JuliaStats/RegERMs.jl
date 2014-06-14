export SVM

immutable SVM <: RegERM
    X::Matrix  # n x m matrix of n m-dimensional training examples
    y::Vector  # 1 x n vector with training classes
    λ::Float64 # regularization parameter
    n::Int     # number of training examples
    m::Int     # number of features
end

function SVM(X::Matrix, y::Vector, λ::Float64)
	(n, m) = size(X)
	if (n != length(y))
		throw(DimensionMismatch("Dimensions of X and y mismatch."))
	end
	if (sort(unique(y)) != [-1,1])
		throw(ArgumentError("Class labels have to be either -1 or 1"))
	end
	SVM(X, y, λ, n, m)
end

modelname(::SVM) = "Support Vector Machine"
losses(svm::SVM, w::Vector) = Hinge(w, svm.X, svm.y)
losses(svm::SVM, w::Vector, i::Int) = Hinge(w, svm.X[i,:], [svm.y[i]])
regularizer(svm::SVM, w::Vector) = L2reg(w, svm.λ)
