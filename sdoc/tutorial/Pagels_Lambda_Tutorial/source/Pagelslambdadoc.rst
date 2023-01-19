.. _gettingStart:

Summarize phylogenetic signal module documentation
==================================================

NAME
----
   ete3.tools.ete_PagelsLambda.py

FUNCTIONS
---------
   * Brownian_motion_likelihood(X, :math:`Z0`, deltaSquare, C)
      **Description**:
         | Computing the likelihood of obtaining the data under Brownian motion model by implementing this formula:
         | 
         |   :math:`\text{$L$}(\mathbf{x} | \bar{z}(0), \sigma^2, \mathbf{C}) = \frac{e^{-1/2 (\mathbf{x}-\bar{z}(0) \mathbf{1})^\intercal (\sigma^2 \mathbf{C})^{-1} (\mathbf{x}-\bar{z}(0) \mathbf{1})}} {\sqrt{(2 \pi)^n det(\sigma^2 \mathbf{C})}}`    (Harmon, 2019)
         |  
      **Parameters**:
         | *X*: (the n x 1 vector of trait values), as numpy array
         | :math:`Z0`: (the root state for the character), as numpy.array([[value]])
         | *deltaSquare*: (the rate of evolution), as numpy.array([[value]])
         | *C*: (the n x n variance-covariance matrix under Brownian motion), as numpy array
         |     ** n is the number of taxa in the tree
         |     ** *X* is an n x 1 vector of trait values with species in the same order as *C*

      **Returns**:
         | float: computed Brownian motion likelihood
         | 
   * Ln_Brownian_motion_likelihood(X, :math:`Z0`, deltaSquare, C)
      **Description**:
         | Computing the natural logarithm of the likelihood of obtaining the data under Brownian motion model by implementing this formula:
         | 
         |   :math:`\ln{\text{$L$}(\mathbf{x} | \bar{z}(0), \sigma^2, \mathbf{C})}= -\frac{1}{2} (\mathbf{x}-\bar{z}(0) \mathbf{1})^\intercal (\sigma^2 \mathbf{C})^{-1} (\mathbf{x}-\bar{z}(0) \mathbf{1}) - \frac{1}{2} [n\ln{2 \pi} + \ln{det(\sigma^2 \mathbf{C})}]`
         | 
      **Parameters**:
         | *X*: (the n x 1 vector of trait values), as numpy array
         | :math:`Z0`: (the root state for the character), as numpy.array([[value]])
         | *deltaSquare*: (the rate of evolution), as numpy.array([[value]])
         | *C*: (the n x n variance-covariance matrix under Brownian motion), as numpy array
         |     ** n is the number of taxa in the tree
         |     ** *X* is an n x 1 vector of trait values with species in the same order as *C*
 
      **Returns**:
         | numpy.float64: computed natural logarithm of the Brownian motion likelihood
         | 
   * Brownian_motion_maximumlikelihood(X, C)
      **Description**:
         | Calculating the maximum-likelihood estimates for :math:`\sigma^2` and :math:`\bar{z}(0)` by implemeting following formulas (Harmon, 2019):
         | 
         |   :math:`\hat{\bar{z}}(0) = (\mathbf{1}^\intercal \mathbf{C}^{-1} \mathbf{1})^{-1} (\mathbf{1}^\intercal \mathbf{C}^{-1} \mathbf{x})`
         | 
         |   :math:`\hat{\sigma}_{ML}^2 = \frac{(\mathbf{x} - \hat{\bar{z}}(0) \mathbf{1})^\intercal \mathbf{C}^{-1} (\mathbf{x} - \hat{\bar{z}}(0) \mathbf{1})}{n}`
         | 
      **Parameters**:
         | *X*: (the n x 1 vector of trait values), as numpy array
         | *C*: (the n x n variance-covariance matrix under Brownian motion), as numpy array
         |     ** n is the number of taxa in the tree
         |     ** *X* is an n x 1 vector of trait values with species in the same order as *C*

      **Returns**:
         | numpy.ndarray: the estimated root state for the character
         | numpy.ndarray: the estimated net rate of evolution
         | 
   * lambdaCovarience(C, lambdaVal)
      **Description**:
         | Based on Pagel's :math:`\lambda` to transform the variance-covariance matrix.
         | 
         | Here is a graph to show this transformation (Harmon, 2019):
         |  Orignal matrix: :math:`\mathbf{C_o}=\begin{bmatrix}\sigma_{1}^2 & \sigma_{12} & \dots & \sigma_{1n}\\\sigma_{21} & \sigma_{2}^2 & \dots & \sigma_{2n}\\\vdots & \vdots & \ddots & \vdots\\\sigma_{n1} & \sigma_{n2} & \dots & \sigma_{n}^2\\\end{bmatrix}`
         |     
         |  Transformed matrix: :math:`\mathbf{C_\lambda} =\begin{bmatrix}\sigma_{1}^2 & \lambda \cdot \sigma_{12} & \dots & \lambda \cdot \sigma_{1n}\\\lambda \cdot \sigma_{21} & \sigma_{2}^2 & \dots & \lambda \cdot \sigma_{2n}\\\vdots & \vdots & \ddots & \vdots\\\lambda \cdot \sigma_{n1} & \lambda \cdot \sigma_{n2} & \dots & \sigma_{n}^2\\\end{bmatrix}`
         | 
      **Parameters**:
         | *C*: (the n x n variance-covariance matrix), as numpy array
         | *lambdaVal*: (:math:`\lambda` value and it is restricted to values of :math:`0 <= \lambda <= 1`), as float

      **Returns**:
         | numpy.ndarray: the transformed variance-covariance matrix based on Pagel's :math:`\lambda`
         |  
   * Pagel_lambda_MLE(X, C, lambdaVal)
      **Description**:
         | Based on Pagel's :math:`\lambda`, computing the maximum ln likelihood for given values

      **Parameters**:
         | *X*: (the n x 1 vector of trait values), as numpy array
         | *C*: (the n x n variance-covariance matrix), as numpy array
         | *lambdaVal*: (:math:`\lambda` value and it is restricted to values of :math:`0 <= \lambda <= 1`), as float
         |     ** n is the number of taxa in the tree
         |     ** *X* is an n x 1 vector of trait values with species in the same order as *C*
 
      **Returns**:
         | numpy.float64: computed maximum ln likelihood for given values based on Pagel's :math:`\lambda`
         | numpy.ndarray: the estimated root state for the character
         | numpy.ndarray: the estimated net rate of evolution
         | float: :math:`\lambda` value
         | 
   * Found_Pagel_Maximumlikelihood(X, tree, stepSize, startSearch = 0, EndSearch = 1)
      **Description**:
         | Searching with the given step size and finding out the value of :math:`\lambda` that gives the largest maximum ln likelihood within the searching range 
         |     ** the computation of the maximum ln likelihood for each given lambda is based on Pagel's :math:`\lambda`
 
      **Parameters**:
         | *X*: (the n x 1 vector of trait values), as numpy array
         | *tree*: (the phylogenetic tree), as ete3 Tree
         | *stepSize*: (the searching step size), as float
         | *startSearch*: (the start value of searching, :math:`0 <= startSearch <= 1`), as float
         | *EndSearch*: (the end value of searching, :math:`0 <= EndSearch <= 1`), as float

      **Returns**:
         | numpy.float64: the largest maximum ln likelihood within the searching range
         | float: the value of :math:`\lambda` that gives the largest maximum ln likelihood within the searching range
         | list: storing all maximum ln likelihoods
         | list: storing all lambda values
         | 
   * Covariance(bac_tree)
      **Description**:
         | Computing the covariance matrix for the given phylogenetic tree

      **Parameters**:
         | *bac_tree*: (the phylogenetic tree), as ete3 Tree

      **Returns**:
         | numpy.ndarray: the covariance matrix
         | 

REFERENCES
-----------
   Harmon, L. J. (2019). Phylogenetic comparative methods \. Open Textbook Library. 