export sgd

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
			x = x_previous - alpha*grad(x_previous, i)
		end

		converged = norm(x - x_previous) < xtol
		
		# Maintain a record of previous position
        copy!(x_previous, x)
    end
    x
end