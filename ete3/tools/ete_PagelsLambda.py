import numpy as np
import math
from numpy.linalg import det, pinv, slogdet
from ete3 import Tree

np.seterr(over='raise')

'''Following functions is used for input checkings: they will be used to check whether function inputs are appropriate'''

def Z0_delatSquare_basic_Unittest(val, text):
    randomMatrix = np.array([[1]])
    randomNumpyInt = np.array([1])[0]
    randomNumpyFloat = np.array([11.1])[0]
    if type(val) != type(randomMatrix):
        raise TypeError(f"{text} should be a numpy array")
    if len(val.shape) != 2:
        raise ValueError(f"{text} should be np.array([[value]])")
    if val.shape[0] != 1 or val.shape[1] != 1:
        raise ValueError(f"{text} should be np.array([[value]])")
    if type(val[0][0]) != type(randomNumpyInt) and type(val[0][0]) != type (randomNumpyFloat):
        raise TypeError(f"Value in {text} sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")

# x is an n x 1 vector of trait
def Basic_unittest(X,Z0, deltaSquare, C):
    randomMatrix = np.array([[1]])
    randomNumpyInt = np.array([1])[0]
    randomNumpyFloat = np.array([11.1])[0]
    # X OK
    if type(X) != type(randomMatrix):
        raise TypeError("X should be a numpy array")
    if len(X.shape) != 2:
        raise ValueError("X should be an n x 1 vector")
    if X.shape[1] != 1:
        raise ValueError("X should be an n x 1 vector")
    for x in X:
        j = x[0]
        if type(j) != type(randomNumpyInt) and type(j) != type(randomNumpyFloat):
            raise TypeError(f"Value in X sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")
    # Z0 OK
    Z0_delatSquare_basic_Unittest(Z0, "Z0")
    
    # deltaSquare OK
    Z0_delatSquare_basic_Unittest(deltaSquare, "deltaSquare")
    if deltaSquare[0][0] < 0:
        raise ValueError("deltaSquare should be greater than or equal to 0")
        
    # C OK
    if type(C) != type(randomMatrix):
        raise TypeError("C should be a numpy array")
    if len(C.shape) != 2:
        raise ValueError("C should be an n x n symmetric matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C should be an n x n symmetric matrix")
    if not np.array_equal(C, C.T):
        raise ValueError("C should be an n x n symmetric matrix")
    for c in C:
        for j in c:
            if type(j) != type(randomNumpyInt) and type(j) != type(randomNumpyFloat):
                raise TypeError(f"Value in C sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")
    if X.shape[0] != C.shape[0]:
        raise ValueError("X should be an n x 1 vector and C should be an n x n symmetric matrix")
# Finished

def Basic_unittest_X_C(X, C):
    randomMatrix = np.array([[1]])
    randomNumpyInt = np.array([1])[0]
    randomNumpyFloat = np.array([11.1])[0]
    # X OK
    if type(X) != type(randomMatrix):
        raise TypeError("X should be a numpy array")
    if len(X.shape) != 2:
        raise ValueError("X should be an n x 1 vector")
    if X.shape[1] != 1:
        raise ValueError("X should be an n x 1 vector")
    for x in X:
        j = x[0]
        if type(j) != type(randomNumpyInt) and type(j) != type(randomNumpyFloat):
            raise TypeError(f"Value in X sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")        
    # C OK
    if type(C) != type(randomMatrix):
        raise TypeError("C should be a numpy array")
    if len(C.shape) != 2:
        raise ValueError("C should be an n x n symmetric matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C should be an n x n symmetric matrix")
    if not np.array_equal(C, C.T):
        raise ValueError("C should be an n x n symmetric matrix")
    for c in C:
        for j in c:
            if type(j) != type(randomNumpyInt) and type(j) != type(randomNumpyFloat):
                raise TypeError(f"Value in C sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")
    if X.shape[0] != C.shape[0]:
        raise ValueError("X should be an n x 1 vector and C should be an n x n symmetric matrix")
# Finished

def Basic_unittest_C(C):
    randomMatrix = np.array([[1]])
    randomNumpyInt = np.array([1])[0]
    randomNumpyFloat = np.array([11.1])[0]    
    # C OK
    if type(C) != type(randomMatrix):
        raise TypeError("C should be a numpy array")
    if len(C.shape) != 2:
        raise ValueError("C should be an n x n symmetric matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C should be an n x n symmetric matrix")
    if not np.array_equal(C, C.T):
        raise ValueError("C should be an n x n symmetric matrix")
    for c in C:
        for j in c:
            if type(j) != type(randomNumpyInt) and type(j) != type(randomNumpyFloat):
                raise TypeError(f"Value in C sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")
# Finished

def Basic_unittest_X(X):
    randomMatrix = np.array([[1]])
    randomNumpyInt = np.array([1])[0]
    randomNumpyFloat = np.array([11.1])[0]
    # X OK
    if type(X) != type(randomMatrix):
        raise TypeError("X should be a numpy array")
    if len(X.shape) != 2:
        raise ValueError("X should be an n x 1 vector")
    if X.shape[1] != 1:
        raise ValueError("X should be an n x 1 vector")
    for x in X:
        j = x[0]
        if type(j) != type(randomNumpyInt) and type(j) != type(randomNumpyFloat):
            raise TypeError(f"Value in X sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")
# Finished

'''Functions for input checkings end here'''

'''We can calculate the likelihood of obtaining
the data under our Brownian motion model using a standard formula for the
likelihood of drawing from a multivariate normal distribution (Harmon & Open Textbook Library, 2019)'''
# Reference is:
# Harmon, L. J. & Open Textbook Library. (2019). Phylogenetic Comparative Methods. https://open.umn.edu/opentextbooks/textbooks/691 
# Section 4.3: Estimating rates using maximum likelihood (eq. 4.5)
# Since it is an implementation of a mathematic formular, please compare our implementation with that mathematic formular in the reference.
def Brownian_motion_likelihood(X,Z0, deltaSquare, C):
    Basic_unittest(X,Z0, deltaSquare, C)
    X = X * 1.0
    Z0 = Z0 * 1.0
    deltaSquare = deltaSquare * 1.0
    C = C * 1.0
    # one is n x 1 vector of 1
    one = np.full((len(X), 1), 1)
    Z0_vector = Z0 * one
    XSubZ0_vector = X - Z0_vector
    # This is because pinv returns the inverse of your matrix when it is available and the pseudo inverse when it isn't.
    temp = np.dot(np.dot(np.transpose(XSubZ0_vector), pinv(deltaSquare * C)), XSubZ0_vector)
    numerator = math.exp(-1/2 * temp) # This is correct
    try:
        denominator = math.sqrt(((2 * math.pi) ** len(X)) * det(deltaSquare * C)) # This is correct
    except FloatingPointError: # OK to have this error and continue executing next time
        return 0
    likelihood = numerator / denominator
    return likelihood # correct

# ln of Brownian motion likelihood
def Ln_Brownian_motion_likelihood(X,Z0, deltaSquare, C):
    Basic_unittest(X,Z0, deltaSquare, C)
    X = X * 1.0
    Z0 = Z0 * 1.0
    deltaSquare = deltaSquare * 1.0
    C = C * 1.0
    # one is n x 1 vector of 1
    one = np.full((len(X), 1), 1)
    Z0_vector = Z0 * one
    XSubZ0_vector = X - Z0_vector
    # This is because pinv returns the inverse of your matrix when it is available and the pseudo inverse when it isn't.
    temp = np.dot(np.dot(np.transpose(XSubZ0_vector), pinv(deltaSquare * C)), XSubZ0_vector)
    front = -1/2 * temp
    end = 1/2 * (len(X)*np.log(2 * math.pi) + slogdet(deltaSquare * C)[1])
    return (front - end)[0][0]


'''In this case, the maximum-likelihood estimate for each of these two parameters can be calculated analytically (Harmon & Open Textbook Library, 2019)'''
# Reference is:
# Harmon, L. J. & Open Textbook Library. (2019). Phylogenetic Comparative Methods. https://open.umn.edu/opentextbooks/textbooks/691 
# Section 4.3: Estimating rates using maximum likelihood (eq. 4.7) and (eq. 4.8)
# Since it is an implementation of mathematic formulars, please compare our implementation with mathematic formulars in the reference.
def Brownian_motion_maximumlikelihood(X, C):
    Basic_unittest_X_C(X, C)
    X = X * 1.0
    C = C * 1.0
    # one is n x 1 vector of 1
    one = np.full((len(X), 1), 1)
    z0hat_front = pinv(one.T @ pinv(C) @ one)
    z0hat_end = one.T @ pinv(C) @ X
    # estimated root state for the character
    z0hat = z0hat_front * z0hat_end

    # maximum likelihood delta square
    numerator = (X - z0hat * one).T @ pinv(C) @ (X - z0hat * one)
    denominator = len(X)
    # estimated net rate of evolution
    deltaSquarehat = numerator / denominator
    return z0hat, deltaSquarehat

# Based on Pagel's lambda to transform the phylogenetic variance-covariance matrix.
# compresses internal branches while leaving the tip branches of the tree unaffected
# Reference is:
# Harmon, L. J. & Open Textbook Library. (2019). Phylogenetic Comparative Methods. https://open.umn.edu/opentextbooks/textbooks/691 
# Section 6.2: Transforming the evolutionary variancecovariance matrix (Equation 6.2)
# Since it is an implementation of a mathematic formular, please compare our implementation with that mathematic formular in the reference.
def lambdaCovarience(C, lambdaVal):
    Basic_unittest_C(C)
    if type(lambdaVal) != type(1) and type(lambdaVal) != type(0.5):
        raise TypeError("lambdaVal should be either an integer or a float")
    # 0 <= lambda <= 1
    if (lambdaVal < 0) or (lambdaVal > 1):
        raise ValueError("Lambda value: 0 <= lambda <= 1")
    C = C * 1.0
    n = len(C)
    for i in range(0,n):
        for j in range(0,n):
            # Off diagonal times lambda
            if i != j:
                C[i][j] = C[i][j] * lambdaVal
    return C

# Compute MLE for a given lambda value
def Pagel_lambda_MLE(X, C, lambdaVal):
    Basic_unittest_X_C(X, C)
    if type(lambdaVal) != type(1) and type(lambdaVal) != type(0.5):
        raise TypeError("lambdaVal should be either an integer or a float")
    # 0 <= lambda <= 1
    if (lambdaVal < 0) or (lambdaVal > 1):
        raise ValueError("Lambda value: 0 <= lambda <= 1")
    X = X * 1.0
    C = C * 1.0
    # Compute new covarience matrix
    C_lambda = lambdaCovarience(C, lambdaVal)
    z0hat, deltaSquarehat = Brownian_motion_maximumlikelihood(X, C_lambda)
    # Compute ln likelihood
    Pagel_likelihood = Ln_Brownian_motion_likelihood(X,z0hat,deltaSquarehat, C_lambda)

    return Pagel_likelihood, z0hat, deltaSquarehat, lambdaVal

# Searching with different step sizes. Finding out lambda's value to maximize likelihood
def Found_Pagel_Maximumlikelihood(X, tree, stepSize, startSearch = 0, EndSearch = 1):
    Basic_unittest_X(X)
    # Just used to get type
    defaultTree = Tree("(A:1,(B:1,(C:1,D:1):0.5):0.5);")
    if type(tree) != type(defaultTree):
        raise TypeError(f"tree type should be {type(defaultTree)}")
    treeLen = len(tree)
    if X.shape[0] != treeLen:
        raise ValueError("Wrong X and tree input, dimension does not match")
    # stepSize check
    if type(stepSize) != type(1) and type(stepSize) != type(0.5):
        raise TypeError("stepSize should be either an integer or a float")
    if (stepSize <= 0):
        raise ValueError("stepSize value: 0 < stepSize")
    # startSearch check
    if type(startSearch) != type(1) and type(startSearch) != type(0.5):
        raise TypeError("startSearch should be either an integer or a float")
    if (startSearch < 0) or (startSearch > 1):
        raise ValueError("startSearch value: 0 <= startSearch <= 1")
    # EndSearch check
    if type(EndSearch) != type(1) and type(EndSearch) != type(0.5):
        raise TypeError("EndSearch should be either an integer or a float")
    if (EndSearch < 0) or (EndSearch > 1):
        raise ValueError("EndSearch value: 0 <= EndSearch <= 1")
    if startSearch > EndSearch:
        raise ValueError("startSearch shoud be smaller than or equal to EndSearch")
    # OK
    X = X * 1.0

    # Initialization
    lambdaVal = startSearch
    maxlikelihood = -math.inf
    maxlikelihood_lambda = -math.inf
    # Record all likelihood and lambda value
    likelihoodSave = []
    lambdaValSave = []
    C = Covariance(tree)
    # Try different lambda values and try to find its corresponding MLE
    while lambdaVal <= EndSearch:
        tmp_likelihood,tmp_z0hat, tmp_deltaSquarehat, tmp_lambdaVal= Pagel_lambda_MLE(X, C, lambdaVal)
        # If tmp value is larger
        if maxlikelihood < tmp_likelihood:
            maxlikelihood = tmp_likelihood
            maxlikelihood_lambda = lambdaVal
        likelihoodSave.append(tmp_likelihood)
        lambdaValSave.append(lambdaVal)
        lambdaVal += stepSize
    # Return all of them
    return maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave

# This function will return covariance matrix
def Covariance(bac_tree):
    defaultTree = Tree("(A:1,(B:1,(C:1,D:1):0.5):0.5);")
    if type(bac_tree) != type(defaultTree):
        raise TypeError(f"bac_tree type should be {type(defaultTree)}")
    # OK
    # Create n by n matrix Make sure pruning the tree or it maybe too large and have error occur.
    C = np.zeros(shape=(len(bac_tree), len(bac_tree)))
    # Used to tranverse through the matrix
    i_counter = -1
    j_counter = -1
    # Tranverse through all leaves
    for leaf_i in bac_tree:
        # Corresponding index
        i_counter += 1
        # Tranverse through all leaves
        for leaf_j in bac_tree:
            j_counter += 1
            # If they are the same leaf
            if leaf_i == leaf_j:
                # Covariance is just its distance to the root
                C[i_counter][j_counter] = leaf_i.get_distance(bac_tree)
            else:
                # Get their first common ancestor and compute its distance to root
                commonAncestor = leaf_i.get_common_ancestor(leaf_j)
                C[i_counter][j_counter] = commonAncestor.get_distance(bac_tree)
        j_counter = -1
    return C