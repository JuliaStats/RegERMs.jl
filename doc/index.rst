RegERMs.jl
===================================

Regularized empirical risk minimization (RegERM) is a general concept that defines a family of optimization problems in machine learning, as, e.g., Support Vector Machine, Logistic Regression, and Linear Regression. 

Contents:

.. toctree::
   :maxdepth: 2

   api.rst
   methods.rst
   

Let :math:`{\bf x}_i` be a vector of features describing an instance i and :math:`y_i` be its target value. Then, for a given set of n training instances :math:`\{({\bf x}_i,y_i)\}_{i=1}^n` the goal is to find a model :math:`{\bf w}` that minimizes the regularized empirical risk:
	.. math::
		\sum_{i=1}^n \ell({\bf w}, {\bf x}_i, y_i) + \Omega({\bf w}).

The loss function :math:`\ell` measures the disagreement between the true label :math:`y` and the model prediction and the regularizer :math:`\Omega` penalizes the model's complexity.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

