type L1RDASolver <: RegressionSolver end

function solve(model::RegressionModel, 
			   method::RegERM, ::L1RDASolver, 
			   X::AbstractMatrix, y::AbstractVector,
			   tolerance::Float64 = 1e-8,
			   iterations::Int = 1000)
	# find the dimensions 
	(N,d) = size(X)

	# define a derivative for a loss function
	function loss_grad(theta::Vector, i::Int)
        derivs(loss(method), values(model, X[i,:], theta), [y[i]])
    end

    # Maintain current state in w, previous state in w_previous and initial dual average in g
    w, w_previous, g = copy(model.theta), copy(model.theta), zeros(d)

    for t=1:iterations
    	λ_rda = λ+(ρ*γ)/sqrt(t)
        idx = round(N*rand(k))
        idx[idx .< 1] = 1

        eval = (X[:,idx]*w).*Y[idx]        
        grad = map(i->loss_grad(w,i), idx[find(eval .< 1)])

        # calculate dual average
        g = ((t-1)/t).*g - (1/(t)).*sum(grad)
        
        # find a close form solution
        w = -(sqrt(t)/γ).*(g - λ_rda.*sign(g))
        w[w. <= λ_rda] = 0

        # check the stopping criteria w.r.t. Tolerance
        if vecnorm(w - w_previous) < tolerance
            break
        end
    end

    w
end