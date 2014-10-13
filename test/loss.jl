eps = 1e-5
# list of losses
binomialLosses = [
    HingeLoss(),
    LogisticLoss(),
    SquaredLoss()
]

multinomialLosses = [
    MultinomialLogisticLoss()
]

fv, y = [2.; 4.; 0.; 40; 40], [-1; -1; 1; 1; -1]
# expected values for f and y
expected_values(::HingeLoss) = [3.0, 5.0, 1.0, 0.0, 41.0]
expected_derivs(::HingeLoss) = [1.0, 1.0, -1.0, 0.0, 1.0]

expected_values(::LogisticLoss) = [2.12693, 4.01815, 0.693147, 0.0, 40.0]
expected_derivs(::LogisticLoss) = [0.88079, 0.98201,-0.5, 0.0, 1.0]
expected_values(::MultinomialLogisticLoss) = expected_values(LogisticLoss())
expected_derivs(::MultinomialLogisticLoss) = [1-0.88079, 1-0.98201,-0.5, 0.0, 1-1.0]

expected_values(::SquaredLoss) = [4.5, 12.5, 0.5, 760.5,840.5]
expected_derivs(::SquaredLoss) = [3.0, 5.0, -1.0, 39.0, 41.0]

for loss in binomialLosses
    print(" - ")
    println(loss)

    # check values
    @test_approx_eq_eps values(loss, fv, y) expected_values(loss) eps
    for i in 1:3
        @test_approx_eq_eps value(loss, fv[i], y[i]) expected_values(loss)[i] eps
    end
    @test_approx_eq_eps tloss(loss, fv, y) sum(expected_values(loss)) eps

    # check derivatives
    @test_approx_eq_eps derivs(loss, fv, y) expected_derivs(loss) eps
    for i in 1:3
        @test_approx_eq_eps deriv(loss, fv[i], y[i]) expected_derivs(loss)[i] eps
    end
    
    # check values and derivatives
    for i in 1:3
        @test_approx_eq_eps [value_and_deriv(loss, fv[i], y[i])...] [expected_values(loss)[i], expected_derivs(loss)[i]] eps
    end
end

# multinomial version for k=2 should be identical to binomial loss
fv_mul, y_mul = fv'', int((-y+3)/2)
for loss in multinomialLosses
    loss = multinomialLosses[1]
    print(" - ")
    println(loss)

    # check values
    @test_approx_eq_eps values(loss, fv_mul, y_mul) expected_values(loss) eps
    for i in 1:3
        @test_approx_eq_eps value(loss, vec(fv_mul[i,:]), y_mul[i]) expected_values(loss)[i] eps
    end
    @test_approx_eq_eps tloss(loss, fv_mul, y_mul) sum(expected_values(loss)) eps

    # check derivatives
    @test_approx_eq_eps derivs(loss, fv_mul, y_mul) expected_derivs(loss) eps

end
