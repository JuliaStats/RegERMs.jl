using RegERMs, Base.Test

# Seed random generator to avoid random effects in testing
srand(1)

X = [1 1; 2 2;  1 -1]
y = [-1; -1; 1]

@test_approx_eq_eps optimize(SVM(X, y), optimizer=:l_bfgs).theta [-0.1499984861841396,-0.3499984861803246] 1e-5
@test_approx_eq_eps optimize(SVM(X, y; λ=1.0), optimizer=:l_bfgs).theta [1.34394e-16,-1.0] 1e-5
@test_approx_eq_eps optimize(SVM(X, y; λ=10.1), optimizer=:l_bfgs).theta [1.34394e-16,-1.0] 1e-5
@test_approx_eq_eps optimize(SVM(X, y), optimizer=:sgd).theta [-0.1499984861841396,-0.3499984861803246] 5e-2
@test_approx_eq_eps optimize(SVM(X, y; λ=1.0), optimizer=:sgd).theta [1.34394e-16,-1.0] 5e-2
@test_approx_eq_eps optimize(SVM(X, y; λ=10.0), optimizer=:sgd).theta [1.34394e-16,-1.0] 5e-2
@test_approx_eq_eps optimize(SVM(X, y, L1RDAParameters(0.1,1.0,1.0)), optimizer=:l1_rda).theta [-14.811388300841895,-37.39005079444413] 5e-2
@test_approx_eq_eps optimize(SVM(X, y, L1RDAParameters(1.0,1.0,1.0)), optimizer=:l1_rda).theta [0.0,-8.866306299725341] 5e-2
@test_approx_eq_eps optimize(SVM(X, y, L1RDAParameters(10.0,1.0,1.0)), optimizer=:l1_rda).theta [0.0,0.0] 5e-2
model1 = optimize(SVM(X, y, L1RDAParameters(0.1,1.0,1.0)), optimizer=:l1_rda)
model2 = optimize(SVM(X, y, λ=10.0), optimizer=:sgd)

@test predict(model1, X) == y
@test predict(model2, X) == y
show(IOBuffer(), SVM(X, y))

@test_throws DimensionMismatch SVM(X', y) 
@test_throws ArgumentError SVM(X, [3; 3; 2])
@test_throws ArgumentError optimize(SVM(X, y, L1RDAParameters(-0.1,1.0,1.0)), optimizer=:l1_rda)
@test_throws ArgumentError optimize(SVM(X, y, L1RDAParameters(0.1,-1.0,1.0)), optimizer=:l1_rda)
@test_throws ArgumentError optimize(SVM(X, y, L1RDAParameters(0.1,1.0,-1.0)), optimizer=:l1_rda)