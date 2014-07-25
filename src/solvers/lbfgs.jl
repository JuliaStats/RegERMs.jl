type LBFGSSolver <: RegressionSolver end

# TODO: λ should be a part of the regularizer and be handled at the model level (e.g., via automatic cross-validation if not provided)

function solve(model::RegressionModel, method::RegERM, ::LBFGSSolver, X::AbstractMatrix, y::AbstractVector, λ::Float64)
    function tloss_grad(theta::Vector)
      n = size(X,1)

      grad_model = gradient(model.f, X[1,:], theta)
      grad_loss = derivs(loss(method), values(model, X[1,:], theta), [y[1]])

      total = broadcast(*, grad_loss',grad_model)
      for i = 2:n
        grad_model = gradient(model.f, X[i,:], theta)
        grad_loss = derivs(loss(method), values(model, X[i,:], theta), [y[i]])

        total += broadcast(*, grad_loss',grad_model)
      end
      return vec(total)
    end

    reg_grad(theta::Vector) = gradient(regularizer(method, theta, λ))
    obj(theta::Vector) = objective(method, model, λ, theta)
    grad!(theta::Vector, storage::Vector) = storage[:] = tloss_grad(theta) + reg_grad(theta)

    Optim.optimize(obj, grad!, model.theta, method=:l_bfgs, linesearch! = Optim.interpolating_linesearch!).minimum
end