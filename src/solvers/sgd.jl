type SGDSolver <: RegressionSolver end

# TODO: λ should be a part of the regularizer and be handled at the model level (e.g., via automatic cross-validation if not provided)
function solve(model::RegressionModel, method::RegERM, ::SGDSolver, X::AbstractMatrix, y::AbstractVector, λ::Float64)
    function loss_grad(theta::Vector, i::Int)
        grad_model = gradient(model.f, X[i,:], theta)
        grad_loss = derivs(loss(method), values(model, X[i,:], theta), [y[i]])
        vec(broadcast(*, grad_loss',grad_model))
    end

    reg_grad(theta::Vector) = gradient(regularizer(method, theta, λ)) / method.n
    grad(theta::Vector, i::Int) = loss_grad(theta, i) + reg_grad(theta)

    sgd(grad, method.n, model.theta)
end

# step size has to statisfy the conditions sum alpha^2 < inf and sum alpha = inf
# 1 / k and 1 / sqrt(k) are common choicses
sqrt_step_size(k::Int) = 1 / sqrt(k)
lin_step_size(k::Int) = 1 / k

function sgd{T}(grad::Function, 
                n::Int, 
                initial_x::Array{T};
                iterations::Int=1000,
                xtol::Real = 1e-8,
                step_size::Function = sqrt_step_size)

    # Maintain current state in x and previous state in x_previous
    x, x_previous = copy(initial_x), copy(initial_x)

    # Count the total number of iterations
    iteration = 0

    # Count the total number of updates
    k = 0

    # Iterate until convergence
    converged = false
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        # randomly iterate over gradients
        for i = randperm(n)
            k += 1

            # Update step size
            alpha = step_size(k)

            # Update current solution
            x -= alpha*grad(x, i)
        end

        converged = norm(x - x_previous) < xtol
        
        # Maintain a record of previous position
        copy!(x_previous, x)
    end
    x
end