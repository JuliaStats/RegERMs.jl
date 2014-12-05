type L1RDASolver <: RegressionSolver end

function solve(model::RegressionModel, 
			   method::RegERM, ::L1RDASolver, 
			   X::AbstractMatrix, y::AbstractVector,
			   tolerance::Float64 = 1e-8, k::Int=1,
			   iterations::Int = 1000)
    (N,d) = size(X)
    (λ,ρ,γ) = method.params.λ, method.params.ρ, method.params.γ


    # define a derivative for a loss function
    function loss_grad(theta::Vector, i::Int)
        grad_model = gradient(model.f, X[i,:], theta)
        grad_loss = derivs(loss(method), values(model, X[i,:], theta), [y[i]])
        vec(broadcast(*, grad_loss',grad_model))
    end

    # Maintain current state in w, previous state in w_previous and initial dual average in g
    (w, w_previous, g) = copy(model.theta), copy(model.theta), zeros(d)

    for t=1:iterations
        λ_rda = λ+(ρ*γ)/sqrt(t)
        idx = int(ceil(N*rand(k)))

        eval = (X[idx,:]*w).*y[idx]
        grad = map(i->loss_grad(w,i), idx[find(eval .< 1)])

        # calculate dual average
        g = ((t-1)/t).*g - (1/t).*sum(grad)

        # find a close form solution
        w = -(sqrt(t)/γ).*(g - λ_rda.*sign(g))
        w[w .<= λ_rda] = 0

        # check the stopping criteria w.r.t. Tolerance
        if vecnorm(w - w_previous) < tolerance
            break
        end
    end
    -w
end