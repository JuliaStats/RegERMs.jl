abstract Hyperparameters
immutable RegularizationParameters <: Hyperparameters
	λ::Float64
end
immutable L1RDAParameters <: Hyperparameters
	λ::Float64
	γ::Float64
	ρ::Float64
end

function check_hyperparameters(param::RegularizationParameters) 
	if (param.λ <= 0)
        throw(ArgumentError("Regularization parameter λ has to be positive"))
    end
end
function check_hyperparameters(param::L1RDAParameters) 
	if (param.λ <= 0)
        throw(ArgumentError("Regularization parameter λ has to be positive"))
    end
    if (param.γ <= 0)
        throw(ArgumentError("l1-RDA parameter γ has to be positive"))
    end
    if (param.ρ <= 0)
        throw(ArgumentError("l1-RDA parameter ρ has to be positive"))
    end
end