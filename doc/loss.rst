Loss functions
===================================
Loss functions measure the disagreement between the true label :math:`y\in\{-1,1\}` and the prediction.

Loss functions implement the following main methods:

.. function:: value(l::Loss) 

	Compute the value of the loss.

.. function:: gradient(l::Loss) 

	Compute the gradient of the loss.

The following loss functions are implemented:

.. function:: Logistic(w::Vector, X::Matrix, y::Vector)
	
	Return a vector of the logistic loss evaluated for all given training instances :math:`\bf X` and the labels :math:`\bf y`
	
	.. math::
		\ell({\bf w}, {\bf x}, y)&=\log(1+exp(-y{\bf x}^T{\bf w})),

	where :math:`{\bf w}` is the weight vector of the decision function.

	.. note::
		The logistic loss corresponds to a likelihood function under an exponential family assumption of the class-conditional distributions :math:`p({\bf x}|y;{\bf w})`.

.. function:: Squared(w::Vector, X::Matrix, y::Vector)
	
	Return a vector of the squared loss evaluated for all given training instances :math:`\bf X` and the labels :math:`\bf y`
	
	.. math::
		\ell({\bf w}, {\bf x}, y)&=(y-{\bf x}^T{\bf w})^2,

	where :math:`{\bf w}` is the weight vector of the decision function.

.. function:: Hinge(w::Vector, X::Matrix, y::Vector)
	
	Return a vector of the hinge loss evaluated for all given training instances :math:`\bf X` and the labels :math:`\bf y`
	
	.. math::
		\ell({\bf w}, {\bf x}, y)&=\max(0, 1-y{\bf x}^T{\bf w}),

	where :math:`{\bf w}` is the weight vector of the decision function.

	.. note::
		The hinge loss corresponds to a max-margin assumption.