using RegERMs, Base.Test

X = [1 1; 2 2;  1 -1];
y = [-1; -1; 1];

svm = SVM(X, y, 1.0)
@test_approx_eq_eps [optimize(svm)...] [1.34394e-16,-1.0] 1e-5
show(IOBuffer(), svm)