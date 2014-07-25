
abstract Loss

# for discriminative (conditional) models
abstract OrdinalLoss <: Loss
abstract NominalLoss <: Loss

# for binary-class cases
abstract BinomialLoss <: NominalLoss
# for multi-class cases
abstract MultinomialLoss <: NominalLoss

# fall back
value_and_deriv(l::Loss, fv::Real, y::Real) = (value(l, fv, y), deriv(l, fv, y))
value_and_deriv(l::MultinomialLoss, fv::AbstractVector, y::Real) = (value(l, fv, y), deriv(l, fv, y))

function tloss(l::Loss, fv::AbstractVector, y::AbstractVector)
    n = size(fv, 1)  # n is the number of samples
    s = 0.0
    for i = 1:n
        s += value(l, fv[i], y[i])
    end
    return s
end

function tloss(l::MultinomialLoss, fv::AbstractMatrix, y::AbstractVector)
    n = size(fv, 1)  # n is the number of samples
    s = 0.0
    for i = 1:n
        s += value(l, vec(fv[i,:]), y[i])
    end
    return s
end

function values(l::Loss, fv::AbstractVector, y::AbstractVector)
    n = size(fv, 1)  # n is the number of samples
    v = zeros(n)
    for i = 1:n
        v[i] = value(l, fv[i], y[i])
    end
    return v
end

function values(l::MultinomialLoss, fv::AbstractMatrix, y::AbstractVector)
    n = size(fv, 1)  # n is the number of samples
    v = zeros(n)
    for i = 1:n
        v[i] = value(l, vec(fv[i,:]), y[i])
    end
    return v
end

function derivs(l::Loss, fv::AbstractVector, y::AbstractVector)
    n = size(fv, 1)  # n is the number of samples
    dv = zeros(n)
    for i = 1:n
        dv[i] = deriv(l, fv[i], y[i])
    end
    return dv
end

function derivs(l::MultinomialLoss, fv::AbstractMatrix, y::AbstractVector)
    n, k = size(fv)  # n is the number of samples
    dv = zeros(n, k)
    for i = 1:n
        dv[i,:] = deriv(l, vec(fv[i,:]), y[i])
    end
    return dv
end

## logistic loss

# binomial
type LogisticLoss <: BinomialLoss end

value(l::LogisticLoss, fv::Real, y::Int) = -y*fv>34 ? -y*fv : log(1+exp(-y*fv))
deriv(l::LogisticLoss, fv::Real, y::Int) = -y / (1 + exp(y*fv))

function value_and_deriv(::LogisticLoss, fv::Real, y::Int)
    emyfv = exp(-y*fv)
    (fv>34 ? -y*fv : log(1+emyfv), -y * emyfv/(1+emyfv))
end

# multinomial (where fv are the decision values for k-1 difference vectors)
# see http://en.wikipedia.org/wiki/Multinomial_logistic_regression#As_a_set_of_independent_binary_regressions
type MultinomialLogisticLoss <: MultinomialLoss end

value(::MultinomialLogisticLoss, fv::AbstractVector, y::Int) = log(1+sum(exp(fv)))-[fv, 0.0][y]

# TODO(cs): rename to gradient
deriv(l::MultinomialLogisticLoss, fv::Vector, y::Int) = exp(-value(l, fv, y))-(1:length(fv).==y)

## squared loss

type SquaredLoss <: OrdinalLoss end

value(::SquaredLoss, fv::Real, y::Real) = (r = fv-y; 0.5 * r*r)
deriv(::SquaredLoss, fv::Real, y::Real) = fv-y

value_and_deriv(::SquaredLoss, fv::Real, y::Real) = (r=fv-y; (0.5 * r*r, r))

## hinge loss

type HingeLoss <: BinomialLoss end

value(::HingeLoss, fv::Real, y::Int) = max(0, 1-y*fv)
deriv(l::HingeLoss, fv::Real, y::Int) = -y*(value(l, fv, y) > 0)

value_and_deriv(::HingeLoss, fv::Real, y::Int) = (h = max(0, 1-y*fv); (h, -y*(h>0)))
