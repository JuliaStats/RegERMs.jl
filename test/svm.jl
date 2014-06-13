using RegERMs, Base.Test

X = [1 1; 2 2;  1 -1];
y = [-1; -1; 1];

svm = SVM(X,y)
@test_approx_eq [train(svm, 1.0)...] [1.34394e-16,-1.0]