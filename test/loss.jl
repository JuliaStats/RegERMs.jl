w = [1; 1]
X = [1 1; 2 2;  1 -1];
y = [-1; -1; 1];

@test_approx_eq_eps value(Logistic(w, X, y)) [2.12693, 4.01815, 0.693147] 1e-5
@test_approx_eq_eps gradient(Logistic(w, X[1,:], [y[1]])) [0.880797; 0.880797] 1e-5
@test_approx_eq_eps gradient(Logistic(w, X[2,:], [y[2]])) [1.96403; 1.96403] 1e-5
@test_approx_eq_eps gradient(Logistic(w, X[3,:], [y[3]])) [-0.5; 0.5] 1e-5
@test_approx_eq_eps sum(gradient(Logistic(w, X, y)),2) [2.34482; 3.34482] 1e-5

@test_approx_eq_eps value(Squared(w, X, y)) [4.5, 12.5, 0.5] 1e-5
@test_approx_eq_eps gradient(Squared(w, X[1,:], [y[1]])) [3; 3] 1e-5
@test_approx_eq_eps gradient(Squared(w, X[2,:], [y[2]])) [10; 10] 1e-5
@test_approx_eq_eps gradient(Squared(w, X[3,:], [y[3]])) [-1; 1] 1e-5
@test_approx_eq_eps sum(gradient(Squared(w, X, y)),2) [12; 14] 1e-5

@test_approx_eq_eps value(Hinge(w, X, y)) [3, 5, 1] 1e-5
@test_approx_eq_eps gradient(Hinge(w, X[1,:], [y[1]])) [1; 1] 1e-5
@test_approx_eq_eps gradient(Hinge(w, X[2,:], [y[2]])) [2; 2] 1e-5
@test_approx_eq_eps gradient(Hinge(w, X[3,:], [y[3]])) [-1; 1] 1e-5
@test_approx_eq_eps sum(gradient(Hinge(w, X, y)),2) [2; 4] 1e-5
