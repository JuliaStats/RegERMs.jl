using RegERMs, Base.Test

# Seed random generator to avoid random effects in testing
srand(1)

X = [1 1; 2 2;  1 -1];
y = [-1; -1; 1];

@test_approx_eq_eps [optimize(LinReg(X, y, 0.1), method=:closed_form)...] [-0.0666666666666667,-0.23333333333333334] 1e-5
@test_approx_eq_eps [optimize(LinReg(X, y, 1.0), method=:closed_form)...] [0.06060606060606055,-0.606060606060606] 1e-5
@test_approx_eq_eps [optimize(LinReg(X, y, 10.0), method=:closed_form)...] [0.1791607732201792,-0.7732201791607733] 1e-5
@test_approx_eq_eps [optimize(LinReg(X, y, 0.1), method=:sgd)...] [-0.0666666666666667,-0.23333333333333334] 5e-2
@test_approx_eq_eps [optimize(LinReg(X, y, 1.0), method=:sgd)...] [0.06060606060606055,-0.606060606060606] 5e-2
@test_approx_eq_eps [optimize(LinReg(X, y, 10.0), method=:sgd)...] [0.1791607732201792,-0.7732201791607733] 5e-2
@test_approx_eq_eps [optimize(LinReg(X, y, 0.1), method=:l_bfgs)...] [-0.0666666666666667,-0.23333333333333334] 1e-5
@test_approx_eq_eps [optimize(LinReg(X, y, 1.0), method=:l_bfgs)...] [0.06060606060606055,-0.606060606060606] 1e-5
@test_approx_eq_eps [optimize(LinReg(X, y, 10.0), method=:l_bfgs)...] [0.1791607732201792,-0.7732201791607733] 1e-5
show(IOBuffer(), LinReg(X, y, 10.0))

@test_throws DimensionMismatch LinReg(X', y, 1.0) 
@test_throws ArgumentError optimize(LinReg(X, y, 1.0), method=:blubs)
@test_throws ArgumentError LinReg(X, y, 0.0) 