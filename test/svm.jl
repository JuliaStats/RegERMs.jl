using RegERMs, Base.Test

# Seed random generator to avoid random effects in testing
srand(1)

X = [1 1; 2 2;  1 -1];
y = [-1; -1; 1];

method = :l_bfgs
@test_approx_eq_eps [optimize(SVM(X, y, 0.1), method)...] [-0.1499984861841396,-0.3499984861803246] 1e-5
@test_approx_eq_eps [optimize(SVM(X, y, 1.0), method)...] [1.34394e-16,-1.0] 1e-5
@test_approx_eq_eps [optimize(SVM(X, y, 10.0), method)...] [1.34394e-16,-1.0] 1e-5
method = :sgd
@test_approx_eq_eps [optimize(SVM(X, y, 0.1), method)...] [-0.1499984861841396,-0.3499984861803246] 5e-2
@test_approx_eq_eps [optimize(SVM(X, y, 1.0), method)...] [1.34394e-16,-1.0] 5e-2
@test_approx_eq_eps [optimize(SVM(X, y, 10.0), method)...] [1.34394e-16,-1.0] 5e-2

show(IOBuffer(), SVM(X, y, 10.0))

@test_throws DimensionMismatch SVM(X', y, 1.0) 
@test_throws ArgumentError SVM(X, [3; 3; 2], 1.0) 
