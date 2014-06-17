Regularizer
===================================

Regularization prevent overfitting and introduce additional information (prior knowledge) to solve an *ill-posed* problem.

Regularizers implement the following main methods:

.. function:: value(r::Regularizer) 

	Compute the value of the regularizer.

.. function:: gradient(r::Regularizer) 

	Compute the gradient of the regularizer.

The following regulizers are implemented:

.. function:: L2reg(w::Vector, λ::Float64)

	Implements an :math:`L^2`-norm regularization of the weight vector ``w`` of the decision function:
	
	.. math::
		\Omega({\bf w})&=\frac{1}{2\lambda}\|{\bf w}\|^2,
	
	where :math:`\lambda` controls the influence of the regularizer.

	.. note::
		The :math:`L^2`-norm regularization corresponds to Gaussian prior assumption of :math:`{\bf w}\sim\mathcal{N}({\bf 0},\lambda{\bf I})`.