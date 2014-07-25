
abstract RegressionFunction

# Linear predictive function

type LinearRegressionFunction <: RegressionFunction
end

values(::LinearRegressionFunction, X::AbstractMatrix, w::Vector) = X*w
gradient(::LinearRegressionFunction, X::AbstractMatrix, ::Vector) = X


# TODO: Dual predictive function