# Seed random generator to avoid random effects in testing
srand(1)

X = [1 1; 2 2;  1 -1]

# binomial
y = [-1; -1; 1]
@test_approx_eq_eps optimize(BinomialLogReg(X, y), 0.1, optimizer=:l_bfgs).theta [-0.07276615319846116,-0.1680076736259885] 5e-5
@test_approx_eq_eps optimize(BinomialLogReg(X, y), 1.0, optimizer=:l_bfgs).theta [-0.16588135026949055,-0.840712964600344] 5e-5
@test_approx_eq_eps optimize(BinomialLogReg(X, y), 10.0, optimizer=:l_bfgs).theta [-0.0745584919313508,-2.20259301054857] 5e-5

# TODO: should be moved to test/sgd.jl
@test_approx_eq_eps optimize(BinomialLogReg(X, y), 0.1, optimizer=:sgd).theta [-0.07276615319846116,-0.1680076736259885] 5e-2
@test_approx_eq_eps optimize(BinomialLogReg(X, y), 1.0, optimizer=:sgd).theta [-0.16588135026949055,-0.840712964600344] 5e-2
@test_approx_eq_eps optimize(BinomialLogReg(X, y), 10.0, optimizer=:sgd).theta [-0.0745584919313508,-2.20259301054857] 5e-2
model = optimize(BinomialLogReg(X, y), 10.0, optimizer=:sgd)
@test predict(model, X) == y
show(IOBuffer(), BinomialLogReg(X, y))

@test_throws DimensionMismatch BinomialLogReg(X', y)
@test_throws ArgumentError BinomialLogReg(X, [3; 3; 2])
@test_throws ArgumentError optimize(BinomialLogReg(X, y), 0.0)

# multinomial
y = [1; 1; 2]
method = MultinomialLogReg(X, y)
model = optimize(method, 10.0, optimizer=:l_bfgs)
@test predict(model, X) == y
