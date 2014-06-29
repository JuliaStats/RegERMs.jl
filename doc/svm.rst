Support Vector Machine
===================================

Support vector machines model the relationship between an input variable :math:`{\bf x}` and a continous output variable :math:`y` by finding a hyperplane separating examples belonging to different classes with maximal margin.

.. function:: SVM(X::Matrix, y::Vector; kernel::Symbol=:linear)

	Initialize an SVM object with a data matrix :math:`{\bf X} \in \mathbb{R}^{n\times m}`, a label binary label vector :math:`{\bf y} \in \mathbb{R}^{n}` of :math:`n` :math:`m`-dimensional examples, and a kernel function.

Implements: ``optimize``