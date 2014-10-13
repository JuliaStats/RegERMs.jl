
abstract RegressionModel

function Model(X::AbstractMatrix, y::AbstractVector, regression_type::Symbol, kernel::Symbol)
    if kernel == :linear
        Model(X, y, regression_type)
    else
        error("kernel=$(kernel) is not implemented yet")
    end
end

function Model(X::AbstractMatrix, y::AbstractVector, regression_type::Symbol, f::RegressionFunction=LinearRegressionFunction())
    if regression_type==:ordinal
        w0 = vec(mean(X,1))
        OrdinalModel(f, w0)
    elseif regression_type==:binomial
        w0 = vec(mean(X[y.==1,:],1).-mean(X[y.==-1,:]))
        BinomialModel(f, w0)
    elseif regression_type==:multinomial
        Y = unique(y)
        k = length(Y)
        n, m = size(X)
        w0 = zeros(m,k-1)
        w0_k = mean(X[y.==Y[k],:],1)
        for i in 1:k-1
            w0[:, i] = mean(X[y.==Y[i],:],1)-w0_k
        end
        MultinomialModel(f, vec(w0), k)
    else
        error("regression_type=$(regression_type) is not implemented yet")
    end
end

values(model::RegressionModel, X::AbstractMatrix, theta::AbstractVector=model.theta) = values(model.f, X, theta)

# Ordinal regression

type OrdinalModel <: RegressionModel
    f::RegressionFunction
    theta::Vector            # parameter vector of regression function
end

predict(model::OrdinalModel, X::AbstractMatrix) = values(model.f, X)

# binomial regression

type BinomialModel <: RegressionModel
    f::RegressionFunction
    theta::Vector            # parameter of regression function
end

predict(model::BinomialModel, X::AbstractMatrix) = sign(values(model, X))

# multinomial regression

type MultinomialModel <: RegressionModel
    f::RegressionFunction
    theta::Vector    # k-1 stacked parameter vectors of regression function
    k::Int           # number of classes
end
theta(model::MultinomialModel) = reshape(model.theta, length(model.theta) / (model.k-1), model, model.k-1)

function values(model::MultinomialModel, X::AbstractMatrix, theta::AbstractVector=model.theta)
    n = size(X, 1)
    v = zeros(n, model.k-1)
    m = length(theta)/(model.k-1)
    for k = 1:model.k-1
        v[:,k] = values(model.f, X, theta[(k-1)*m+1:k*m])
    end
    return v
end
predict(model::MultinomialModel, X::AbstractMatrix) = (v = values(model, X); n=size(X,1); [indmax([v[i,:] 0]) for i in 1:n])

