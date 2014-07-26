abstract Regularizer

## l2-regularizer

immutable L2reg <: Regularizer
    w::Vector
    λ::Float64
end

function value(r::L2reg) 
    norm(r.w)^2 / (2 * r.λ)
end

function gradient(r::L2reg) 
    r.w / r.λ
end
