<img src="http://bigcrunsh.github.io/images/logo.png" alt="RegERMs Logo" width="210" height="125"></img>

RegERMs.jl
==========
[![Build Status](https://travis-ci.org/JuliaStats/RegERMs.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/RegERMs.jl)
[![Coverage Status](https://img.shields.io/coveralls/JuliaStats/RegERMs.jl.svg)](https://coveralls.io/r/JuliaStats/RegERMs.jl)

This package implements several machine learning algorithms in a regularised empirical risk minimisation framework (SVMs,  LogReg, Linear Regression) in Julia.

## Quick start

Some examples:

```julia
# define some toy data (XOR - example)
np = 100
nn = 100
X = [randn(int(np/2),1)+1 randn(int(np/2),1)+1; randn(int(np/2-0.5),1)-1 randn(int(np/2-0.5),1)-1;
     randn(int(nn/2),1)+1 randn(int(nn/2),1)-1; randn(int(nn/2-0.5),1)-1 randn(int(nn/2-0.5),1)+1] # examples with 2 features
y = vec([ones(np,1); -ones(nn,1)])       # binary class values

# choose SVM as learning algorithm
svm = SVM(X, y; kernel=:rbf)

# get solution (regularization parameter is 0.1)
regParam = 0.1
model = optimize(svm, regParam)

# make predictions and compute accuracy
ybar = predict(model, X)
acc = mean(ybar .== y)

```

## Documentation

Full documentation available at [Read the Docs](http://regerms.readthedocs.org/en/latest/index.html).

