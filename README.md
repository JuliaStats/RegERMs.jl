RegERMs.jl
==========
[![Build Status](https://travis-ci.org/BigCrunsh/RegERMs.jl.svg?branch=master)](https://travis-ci.org/BigCrunsh/RegERMs.jl)
[![Coverage Status](https://img.shields.io/coveralls/BigCrunsh/RegERMs.jl.png)](https://coveralls.io/r/BigCrunsh/RegERMs.jl)

This package implements several machine learning algorithms in a regularised empirical risk minimisation framework (SVMs,  LogReg, Linear Regression) in Julia.

## Quick start

Some examples:

```julia
# define some toy data
X = [1 1; 2 2;  1 -1]; # (3 examples with 2 features)
y = [-1; -1; 1];       # binary class values for the 3 examples

# choose SVM as learning algorithm (regularization parameter is 0.1)
model = SVM(X, y, 0.1)

# get solution
w = optimize(model)

# make predictions
ybar = sign(X*w)

```

## Documentation

Full documentation available at [Read the Docs](http://regerms.readthedocs.org/en/latest/index.html).

