Ridge Regression
===================================

Ridge regression models the relationship between an input variable :math:`{\bf x}` and a continous output variable :math:`y` by fitting a linear function.

.. function:: LinReg(X::Matrix, y::Vector; kernel::Symbol=:linear)

	Initialize a ridge regression object with a data matrix :math:`{\bf X} \in \mathbb{R}^{n\times m}`, a label binary label vector :math:`{\bf y} \in \mathbb{R}^{n}` of :math:`n` :math:`m`-dimensional examples, and a kernel function.

Implements: ``optimize``