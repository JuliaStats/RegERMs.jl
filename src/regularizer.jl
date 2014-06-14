# l2-regularizer; corresponds to Gaussian prior assumption with mean w0 and variance λ²
l2reg{T<:Real}(w::Vector{T}, λ::Float64) = (norm(w)^2/(2*λ), w./λ)
l2reg{T<:Real, S<:Real}(w::Vector{T}, λ::Float64, w0::Vector{S}) = (norm(w-w0)^2/(2*λ), (w-w0)./λ)