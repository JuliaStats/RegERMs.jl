export Loss, Logistic, Squared, Hinge, value, gradient

abstract Loss

## logistic loss: correspond to likelihood under an exponential family assumption

immutable Logistic <: Loss
    w::Vector
    X::Matrix
    y::Vector
end

function value(l::Logistic) 
    f = l.X*l.w
    log(1.+exp(-l.y.*f))
end
function gradient(l::Logistic) 
    f = l.X*l.w
    g = -l.y.*l.X ./ (1 .+ exp(l.y.*f))
    g'
end


## squared loss

immutable Squared <: Loss
    w::Vector
    X::Matrix
    y::Vector
end

function value(l::Squared) 
    f = l.X*l.w
    (f - l.y).^2 / 2
end
function gradient(l::Squared) 
    f = l.X*l.w
    g = l.X.*(f - l.y)
    g'
end


## hinge loss: correspond to max-margin assumption

immutable Hinge <: Loss
    w::Vector
    X::Matrix
    y::Vector
end

function value(l::Hinge) 
    f = l.X*l.w
    max(0, 1.-l.y.*f)
end
function gradient(l::Hinge) 
    h = value(l)
    g = -l.y.*l.X.*(h .> 0)
    g'
end
