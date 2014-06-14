using RegERMs, Base.Test

# Seed random generator to avoid random effects in testing
srand(1)

X = [1 1; 2 2;  1 -1];
y = [-1; -1; 1];

method = :l_bfgs
@test_approx_eq_eps [optimize(LogReg(X, y, 0.1), method)...] [-0.07276615319846116,-0.1680076736259885] 5e-5
@test_approx_eq_eps [optimize(LogReg(X, y, 1.0), method)...] [-0.16588135026949055,-0.840712964600344] 5e-5
@test_approx_eq_eps [optimize(LogReg(X, y, 10.0), method)...] [-0.0745584919313508,-2.20259301054857] 5e-5
method = :sgd
@test_approx_eq_eps [optimize(LogReg(X, y, 0.1), method)...] [-0.07276615319846116,-0.1680076736259885] 5e-2
@test_approx_eq_eps [optimize(LogReg(X, y, 1.0), method)...] [-0.16588135026949055,-0.840712964600344] 5e-2
@test_approx_eq_eps [optimize(LogReg(X, y, 10.0), method)...] [-0.0745584919313508,-2.20259301054857] 5e-2

show(IOBuffer(), LogReg(X, y, 10.0))

@test_throws DimensionMismatch LogReg(X', y, 1.0) 
@test_throws ArgumentError LogReg(X, [3; 3; 2], 1.0) 