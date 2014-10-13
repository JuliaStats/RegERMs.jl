# Seed random generator to avoid random effects in testing
srand(1)

X = [1 1; 2 2;  1 -1]
y = [-1; -1; 1]

@test_approx_eq_eps optimize(RidgeReg(X, y), 0.1, optimizer=:closed_form).theta [-0.0666666666666667,-0.23333333333333334] 1e-5
@test_approx_eq_eps optimize(RidgeReg(X, y), 1.0, optimizer=:closed_form).theta [0.06060606060606055,-0.606060606060606] 1e-5
@test_approx_eq_eps optimize(RidgeReg(X, y), 10.0, optimizer=:closed_form).theta [0.1791607732201792,-0.7732201791607733] 1e-5
@test_approx_eq_eps optimize(RidgeReg(X, y), 0.1, optimizer=:sgd).theta [-0.0666666666666667,-0.23333333333333334] 5e-2
@test_approx_eq_eps optimize(RidgeReg(X, y), 1.0, optimizer=:sgd).theta [0.06060606060606055,-0.606060606060606] 5e-2
@test_approx_eq_eps optimize(RidgeReg(X, y), 10.0, optimizer=:sgd).theta [0.1791607732201792,-0.7732201791607733] 5e-2
@test_approx_eq_eps optimize(RidgeReg(X, y), 0.1, optimizer=:l_bfgs).theta [-0.0666666666666667,-0.23333333333333334] 1e-5
@test_approx_eq_eps optimize(RidgeReg(X, y), 1.0, optimizer=:l_bfgs).theta [0.06060606060606055,-0.606060606060606] 1e-5
@test_approx_eq_eps optimize(RidgeReg(X, y), 10.0, optimizer=:l_bfgs).theta [0.1791607732201792,-0.7732201791607733] 1e-5
show(IOBuffer(), RidgeReg(X, y))

@test_throws DimensionMismatch RidgeReg(X', y) 
@test_throws ArgumentError optimize(RidgeReg(X, y), 1.0, optimizer=:blubs)
@test_throws ArgumentError optimize(RidgeReg(X, y), 0.0)
