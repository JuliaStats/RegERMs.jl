type LBFGSSolver <: RegressionSolver end

# TODO: λ should be a part of the regularizer and be handled at the model level (e.g., via automatic cross-validation if not provided)
function solve(method::RegERM, ::LBFGSSolver, X::AbstractMatrix, y::AbstractVector, w0::AbstractVector, λ::Float64)
    reg(w::Vector) = regularizer(method, w, λ)
    # TODO: use tloss_and_gradient for efficiency
    tloss_grad(w::Vector) = vec(sum(X.*derivs(loss(method), X*w, y), 1))
    reg_grad(w::Vector) = gradient(reg(w))
    
    obj(w::Vector) = tloss(loss(method), X*w, y) + value(reg(w))
    grad!(w::Vector, storage::Vector) = storage[:] = tloss_grad(w) + reg_grad(w)

    Optim.optimize(obj, grad!, w0, method=:l_bfgs, linesearch! = Optim.interpolating_linesearch!).minimum
end