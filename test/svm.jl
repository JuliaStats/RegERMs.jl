using RegERMs, Base.Test

X = [1 1; 2 2;  1 -1];
y = [-1; -1; 1];

@test_approx_eq_eps [optimize(SVM(X, y, 0.1))...] [-0.1499984861841396,-0.3499984861803246] 1e-5
@test_approx_eq_eps [optimize(SVM(X, y, 1.0))...] [1.34394e-16,-1.0] 1e-5
@test_approx_eq_eps [optimize(SVM(X, y, 10.0))...] [1.34394e-16,-1.0] 1e-5
show(IOBuffer(), SVM(X, y, 10.0))

@test_throws DimensionMismatch SVM(X', y, 1.0) 
@test_throws ArgumentError SVM(X, [3; 3; 2], 1.0) 
