eps = 1e-5
# list of losses
losslist = [
    HingeLoss(),
    LogisticLoss(),
    SquaredLoss()
]

fv, y = [2.; 4.; 0.], [-1; -1; 1]
# expected values for f and y
expected_values(::HingeLoss) = [3.0, 5.0, 1.0]
expected_derivs(::HingeLoss) = [1.0, 1.0, -1.0]

expected_values(::LogisticLoss) = [2.12693, 4.01815, 0.693147]
expected_derivs(::LogisticLoss) = [0.88079, 0.98201,-0.5]

expected_values(::SquaredLoss) = [4.5, 12.5, 0.5]
expected_derivs(::SquaredLoss) = [3.0, 5.0, -1.0]

for loss in losslist
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
