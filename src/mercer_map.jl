# mercer mapping

export MercerMap, apply, linear, rbf

immutable MercerMap
    X::Matrix           # original reference data
    kernel::Symbol      # kernel function
    K::Matrix           # (centered) kernel matrix of reference data
    d::Vector           # largest eigenvalues of kernel matrix
    V::Matrix           # eigenvectors of kernel matrix
end

# constructing mercer map
function MercerMap(X::Matrix, kernel::Symbol)
    kernelfcn = eval(kernel)
    K = center(kernelfcn(X, X))

    d, V = eig(Symmetric(K))

    # consider dimensions with eigenvalues > 1e-9
    i = d .> 1e-9
    V = V[:,i]
    d = sqrt(d[i])

    MercerMap(X, kernel, K, d, V)
end

# map reference data
apply(map::MercerMap) = map.V*diagm(map.d)
function apply(map::MercerMap, X::Matrix)
    kernelfcn = eval(map.kernel)
    KT = center(map.K, kernelfcn(map.X, X))
    KT*map.V*diagm(1./map.d)
end

# center data to origin in feature space (M. Meila: Data Centering in Feature Space, Eq. 17)
function center(K::Matrix, KT::Matrix)
    m = size(K,1)
    M1 = 1/m*ones(size(K))
    N1 = ones(size(KT)) / m
    KT = KT - K*N1 - M1*KT + M1*K*N1
end
center(K::Matrix) = center(K, K)

# kernel functions
linear(X::Matrix, Y::Matrix) = full(X*Y')
rbf(X::Matrix, Y::Matrix; sigma::Float64=1.0) = full(exp(-dist2(X, Y)/(2*sigma^2)))

function dist2(X::Matrix, Y::Matrix)
    xx = sum(X.*X, 2)
    yy = sum(Y.*Y, 2)
    xy = X*Y'
    abs(repmat(xx, 1, size(yy, 1)) + repmat(yy', size(xx, 1), 1) - 2*xy)
end