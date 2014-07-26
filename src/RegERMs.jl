module RegERMs

using StatsBase
using Optim

import StatsBase: predict
import Optim: optimize 

## Exports

export 
    # loss
    Loss,           # abstract type for all kinds of loss
    LogisticLoss,   # Type to represent logistic loss
    SquaredLoss,    # Type to represent squared loss
    HingeLoss,      # Type to represent hinge loss

    value,              # evaluate a single value
    values,             # evaluate multiple values
    deriv,              # evaluate a single derivative value
    derivs,             # evaluate multiple derivative values
    value_and_deriv,    # jointly evaluate function value and derivative
    tloss,              # the total loss 

    # regularizer
    Regularizer,    # abstract type for all kinds of regularizers
    L2reg,          # Type to represent L2 regularizer

    # model
    Model, 
    PrimalModel, 
    DualModel, 
    predict,

    # models
    SVM,
    RidgeReg,
    LogReg,

    # solvers
    LBFGSSolver,
    SGDSolver,

    # optim
    RegERM, 
    RegressionSolver, 
    optimize

## Source files

# include 
include("loss.jl")
include("regularizer.jl")
include("mercer_map.jl")   ## TODO: need some discussion about the API for this
include("model.jl")
include("optim.jl")

# classification methods
include("models/svm.jl")
include("models/logistic_regression.jl")
# regression methods
include("models/ridge_regression.jl")
# solver
include("solvers/sgd.jl")
include("solvers/lbfgs.jl")

end # module
