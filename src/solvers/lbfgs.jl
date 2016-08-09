type LBFGSSolver <: RegressionSolver end

function solve(model::RegressionModel, method::RegERM, ::LBFGSSolver, X::AbstractMatrix, y::AbstractVector)
    function tloss_grad(theta::Vector)
        n = size(X,1)

        grad_model = gradient(model.f, X[1:1,:], theta)
        grad_loss = derivs(loss(method), values(model, X[1:1,:], theta), [y[1]])

        total = broadcast(*, grad_loss',grad_model)
        for i = 2:n
            grad_model = gradient(model.f, X[i:i,:], theta)
            grad_loss = derivs(loss(method), values(model, X[i:i,:], theta), [y[i]])

            total += broadcast(*, grad_loss', grad_model)
        end
        return vec(total)
    end

    reg_grad(theta::Vector) = gradient(regularizer(method, theta))
    obj(theta::Vector) =
        tloss(loss(method), values(model, method.X, theta), method.y) + value(regularizer(method, theta))
    grad!(theta::Vector, storage::Vector) = storage[:] = tloss_grad(theta) + reg_grad(theta)

    Optim.optimize(obj, grad!,
                   model.theta,
                   method=LBFGS(; linesearch! = Optim.interpolating_linesearch!)).minimum
end
