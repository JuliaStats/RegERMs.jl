# logistic loss: corresponds to likelihood under an exponential family assumption
function logistic{T<:Real}(model::RegERM, w::Vector{T})
    f = model.X*w
    losses  = log(1.+exp(-model.y.*f))
    gradient = -model.y.*model.X ./ (1 .+ exp(model.y.*f))

    (losses, gradient)
end

# squared loss
function squared{T<:Real}(model::RegERM, w::Vector{T})
    f = model.X*w
    losses = norm(f - model.y)^2 / 2
    gradient = f - model.y

    (losses, gradient)
end

# hinge loss
function hinge{T<:Real}(model::RegERM, w::Vector{T})
    f = model.X*w
    losses = max(0, 1.-model.y.*f)
    gradient = -model.y.*(losses .> 0)

    (losses, gradient)
end

