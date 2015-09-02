<img src="http://bigcrunsh.github.io/images/logo.png" alt="RegERMs Logo" width="210" height="125"></img>

RegERMs.jl
==========
[![Build Status](https://travis-ci.org/JuliaStats/RegERMs.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/RegERMs.jl)
[![Coverage Status](https://img.shields.io/coveralls/JuliaStats/RegERMs.jl.svg)](https://coveralls.io/r/JuliaStats/RegERMs.jl)
[![RegERMs](http://pkg.julialang.org/badges/RegERMs_0.3.svg)](http://pkg.julialang.org/?pkg=RegERMs&ver=0.3)
[![RegERMs](http://pkg.julialang.org/badges/RegERMs_0.4.svg)](http://pkg.julialang.org/?pkg=RegERMs&ver=0.4)

This package implements several machine learning algorithms in a regularised empirical risk minimisation framework (SVMs,  LogReg, Linear Regression) in Julia.

## Quick start

Some examples:

```julia
using RegERMs

# define some toy data (XOR - example)
np = 100
nn = 100
X = [randn(int(np/2),1)+1 randn(int(np/2),1)+1; randn(int(np/2-0.5),1)-1 randn(int(np/2-0.5),1)-1;
     randn(int(nn/2),1)+1 randn(int(nn/2),1)-1; randn(int(nn/2-0.5),1)-1 randn(int(nn/2-0.5),1)+1] # examples with 2 features
y = int(vec([ones(np,1); -ones(nn,1)]))       # binary class values

# use rbf kernel by using mercer map
map = MercerMap(X, :rbf)
X = RegERMs.apply(map)

# choose (linear) SVM as learning algorithm with regularization parameter 0.1
svm = SVM(X, y; λ=0.1)

# get a solution 
model = optimize(svm)

# make predictions and compute accuracy
ybar = predict(model, X)
acc = mean(ybar .== y)

```

## Documentation

Full documentation available at [Read the Docs](http://regerms.readthedocs.org/en/latest/index.html).

