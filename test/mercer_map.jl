# kernel
X = [1 1; 2 2;  1 -1]
exp_K = [1.0 0.367879  0.135335; 0.367879 1.0 0.00673795; 0.135335 0.00673795 1.0]
@test_approx_eq_eps rbf(X,X) exp_K 1e-6

# test mercer map
X = [1 1; 2 2;  1 -1]
K_centered = [0.444513 -0.144742 -0.299771; -0.144742 0.530245 -0.385503; -0.299771 -0.385503 0.685274]
V = [0.768623 -0.275472; -0.622878 -0.527911; -0.145746 0.803383]
d = [0.786544; 1.02048]
X_mapped = [0.604556 -0.281114; -0.489921 -0.538723; -0.114635 0.819837]

map = MercerMap(X, :rbf)

@test_approx_eq_eps map.K K_centered 1e-6
@test_approx_eq_eps map.d d 1e-6
@test_approx_eq_eps map.V V 1e-6
@test_approx_eq_eps RegERMs.apply(map) X_mapped 1e-6

X = [1 1; 0 0;  -1 -1]
XT = rand(100,2)
V = [-0.7071067811865475; 0.0; 0.7071067811865477]
d = [2.0]
X_mapped = [-1.414213562373095; 0.0; 1.4142135623730954]

map = MercerMap(X, :linear)
@test_approx_eq_eps map.K linear(X,X) 1e-6 # kernel is already centered
@test_approx_eq_eps map.d d 1e-6
@test_approx_eq_eps map.V V 1e-6
@test_approx_eq_eps RegERMs.apply(map, X) X_mapped 1e-6
RegERMs.apply(map, XT)