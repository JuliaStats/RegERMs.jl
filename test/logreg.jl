using RegERMs, Base.Test

X = [1 1; 2 2;  1 -1];
y = [-1; -1; 1];

logreg = LogReg(X, y, 1.0)
@test_approx_eq_eps [optimize(logreg)...] [-0.16588135026949055,-0.840712964600344] 1e-5
show(IOBuffer(), logreg)