import unittest
from PagelsLambda import *
import numpy as np
import pandas as pd
from ete3 import Tree

# python -m unittest [filename]

# Assistance function
def traitsColumnReturn(df, traits_name):
  traits = list(df.loc[:,f'{traits_name}'])
  X = []
  for i in traits:
      X.append([i])
  X = np.array(X)
  return X

# Assistance function
# Reference: https://github.com/simjoly/CourseComparativeMethods/blob/master/lecture6/PD.pdf
'''
Simjoly. (n.d.). Simjoly/CourseComparativeMethods: Examples of application of comparative methods in R. GitHub. Retrieved December 7, 2022, from https://github.com/simjoly/CourseComparativeMethods 
'''
def Seedplant_sanity_assistance():
  seedplant = Tree("../seedplantData/seedplantsNew.tre", format = 0)
  df = pd.read_csv("../seedplantData/seedplants_Formatted.csv")
  # This works drop all rows with NAN value
  df = df.dropna()
  keep = list(df.loc[:,"Code"])
  # Save the tree befor prune
  seedplantSave = seedplant
  # Prune will modify the original tree (Pruning the tree), now seedplant is pruned
  seedplant.prune(keep, preserve_branch_length=True)
  C = Covariance(seedplant)

  # Preserve column names
  headers = list(df.columns)
  reorder = pd.DataFrame(columns=headers)
  # Reorder the data make them in the order of the tree labels
  for leaf in seedplant:
      row = df.loc[df['Code'] == leaf.name]
      reorder = pd.concat([reorder, row],ignore_index=True)

  ValueSave = []
  # traits maxH
  maxH = list(reorder.loc[:,'maxH'])
  X = []
  for i in maxH:
      X.append([i])
  X = np.array(X)
  maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X, seedplant, 0.01, startSearch=0.0, EndSearch=1)
  ValueSave.append([round(maxlikelihood, 2), round(maxlikelihood_lambda,2)])

  # traits Wd
  X = traitsColumnReturn(reorder, 'Wd')
  maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X, seedplant, 0.01, startSearch=0.0, EndSearch=1)
  ValueSave.append([round(maxlikelihood,2), round(maxlikelihood_lambda,2)])

  # traits Sm
  X = traitsColumnReturn(reorder, 'Sm')
  maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X, seedplant, 0.01, startSearch=0.0, EndSearch=1)
  ValueSave.append([round(maxlikelihood,2), round(maxlikelihood_lambda,2)])

  # traits Shade
  X = traitsColumnReturn(reorder, 'Shade')
  maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X, seedplant, 0.01, startSearch=0.0, EndSearch=1)
  ValueSave.append([round(maxlikelihood,2), round(maxlikelihood_lambda,2)])

  # traits N
  X = traitsColumnReturn(reorder, 'N')
  maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X, seedplant, 0.01, startSearch=0.0, EndSearch=1)
  ValueSave.append([round(maxlikelihood,2), round(maxlikelihood_lambda,2)])
  return np.array(ValueSave)

# Unit test
class Test_Ln_Brownian_motion_likelihood(unittest.TestCase):
  # Test method must start with word "test"
  def test_Ln_Brownian_motion_likelihood(self):
    '''Test value when 
     X is an 3 x 1 vector of trait values for the n tip species in the tree,
     Z0 no constraint, 
     deltaSquare >= 0, 
     C is the 3 x 3 symmetric matrix
     '''
    # Test is computed manually
    self.assertAlmostEqual(Ln_Brownian_motion_likelihood(np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]])),-7.11320796)
    self.assertAlmostEqual(Ln_Brownian_motion_likelihood(np.array([[10],[5],[8]]), np.array([[8]]), np.array([[4]]), np.array([[9, 7, 3],[7, 8, 4],[3, 4, 7]])),-8.40570326)
  
  def test_Ln_Brownian_motion_likelihood_values(self): # ValueError
    # Make sure value errors are raised when necessary
    # self.assertRaises(Error, function, arguments)
    self.assertRaises(ValueError, Ln_Brownian_motion_likelihood, np.array([[2,2],[3,3],[4,4]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(ValueError, Ln_Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2],[3]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(ValueError, Ln_Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4],[5]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(ValueError, Ln_Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 3, 6]]))
    self.assertRaises(ValueError, Ln_Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6],[6, 8],[4, 5]]))
    self.assertRaises(ValueError, Ln_Brownian_motion_likelihood, np.array([[3],[2]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))

  def test_Ln_Brownian_motion_likelihood_types(self): # TypeError
    # Make sure type errors are raised when necessary
    self.assertRaises(TypeError, Ln_Brownian_motion_likelihood, [[3],[2],[1]], np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Ln_Brownian_motion_likelihood, np.array([['a'],[2],[1]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Ln_Brownian_motion_likelihood, np.array([[3],[2],[1]]), [[2]], np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Ln_Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([['a']]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Ln_Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), [[4]], np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Ln_Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), np.array([['a']]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Ln_Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4]]), [[8, 6, 4],[6, 8, 5],[4, 5, 6]])
    self.assertRaises(TypeError, Ln_Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 'a']]))

  def test_Brownian_motion_likelihood(self):
    # Brownian_motion_likelihood(X,Z0, deltaSquare, C)
    '''Test value when
    X is an 3 x 1 vector,
    Z0: no constraint,
    deltaSquare >= 0,
    C is the 3 X 3 symmetric matrix
    '''
    self.assertAlmostEqual(Brownian_motion_likelihood(np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]])), \
      np.exp(Ln_Brownian_motion_likelihood(np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))))
    self.assertAlmostEqual(Brownian_motion_likelihood(np.array([[10],[5],[8]]), np.array([[8]]), np.array([[4]]), np.array([[9, 7, 3],[7, 8, 4],[3, 4, 7]])), \
      np.exp(Ln_Brownian_motion_likelihood(np.array([[10],[5],[8]]), np.array([[8]]), np.array([[4]]), np.array([[9, 7, 3],[7, 8, 4],[3, 4, 7]]))))
  
  def test_Brownian_motion_likelihood_values(self): # ValueError
    self.assertRaises(ValueError, Brownian_motion_likelihood, np.array([[2,2],[3,3],[4,4]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(ValueError, Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2],[3]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(ValueError, Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4],[5]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(ValueError, Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 3, 6]]))
    self.assertRaises(ValueError, Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6],[6, 8],[4, 5]]))
    self.assertRaises(ValueError, Brownian_motion_likelihood, np.array([[3],[2]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))

  def test_Brownian_motion_likelihood_types(self): # TypeError
    self.assertRaises(TypeError, Brownian_motion_likelihood, [[3],[2],[1]], np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Brownian_motion_likelihood, np.array([['a'],[2],[1]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Brownian_motion_likelihood, np.array([[3],[2],[1]]), [[2]], np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([['a']]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), [[4]], np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), np.array([['a']]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4]]), [[8, 6, 4],[6, 8, 5],[4, 5, 6]])
    self.assertRaises(TypeError, Brownian_motion_likelihood, np.array([[3],[2],[1]]), np.array([[2]]), np.array([[4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 'a']]))

  def test_Brownian_motion_maximumlikelihood(self):
    # def Brownian_motion_maximumlikelihood(X, C)
    self.assertAlmostEqual(Brownian_motion_maximumlikelihood(np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))[0][0][0], 5/3)
    self.assertAlmostEqual(Brownian_motion_maximumlikelihood(np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))[1][0][0], 2/9)
  
  def test_Brownian_motion_maximumlikelihood_values(self): # ValueError
    self.assertRaises(ValueError, Brownian_motion_maximumlikelihood, np.array([[2,2],[3,3],[4,4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(ValueError, Brownian_motion_maximumlikelihood, np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 3, 6]]))
    self.assertRaises(ValueError, Brownian_motion_maximumlikelihood, np.array([[3],[2],[1]]), np.array([[8, 6],[6, 8],[4, 5]]))
    self.assertRaises(ValueError, Brownian_motion_maximumlikelihood, np.array([[3],[2]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
  
  def test_Brownian_motion_maximumlikelihood_types(self): # TypeError
    self.assertRaises(TypeError, Brownian_motion_maximumlikelihood, [[3],[2],[1]], np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Brownian_motion_maximumlikelihood, np.array([['a'],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]))
    self.assertRaises(TypeError, Brownian_motion_maximumlikelihood, np.array([[3],[2],[1]]), [[8, 6, 4],[6, 8, 5],[4, 5, 6]])
    self.assertRaises(TypeError, Brownian_motion_maximumlikelihood, np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 'a']]))
  
  def test_lambdaCovarience(self):
    # def lambdaCovarience(C, lambdaVal)
    self.assertAlmostEqual(np.array_equal(lambdaCovarience(np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 0.5), np.array([[8, 3, 2],[3, 8, 2.5],[2, 2.5, 6]])), 1)
  
  def test_lambdaCovarience_values(self): # ValueError
    self.assertRaises(ValueError, lambdaCovarience, np.array([[8, 6, 4],[6, 8, 5],[4, 3, 6]]), 0.5)
    self.assertRaises(ValueError, lambdaCovarience, np.array([[8, 6],[6, 8],[4, 5]]), 0.5)
    self.assertRaises(ValueError, lambdaCovarience, np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), -1)
    self.assertRaises(ValueError, lambdaCovarience, np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 2)

  def test_lambdaCovarience_types(self): # TypeError
    self.assertRaises(TypeError, lambdaCovarience, [[8, 6, 4],[6, 8, 5],[4, 5, 6]], 0.5)
    self.assertRaises(TypeError, lambdaCovarience, np.array([[8, 6, 4],[6, 8, 5],[4, 5, 'a']]), 0.5)
    self.assertRaises(TypeError, lambdaCovarience, np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 'a')
  
  def test_Pagel_lambda_MLE(self):
    # def Pagel_lambda_MLE(X, C, lambdaVal)
    self.assertAlmostEqual(Pagel_lambda_MLE(np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 1)[0], -4.19171282178655)
    self.assertAlmostEqual(Pagel_lambda_MLE(np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 1)[1][0][0], 5/3)
    self.assertAlmostEqual(Pagel_lambda_MLE(np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 1)[2][0][0], 2/9)
    self.assertAlmostEqual(Pagel_lambda_MLE(np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 1)[3], 1)
  
  def test_Pagel_lambda_MLE_values(self): # ValueError
    self.assertRaises(ValueError, Pagel_lambda_MLE, np.array([[2,2],[3,3],[4,4]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 0.5)
    self.assertRaises(ValueError, Pagel_lambda_MLE, np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 3, 6]]), 0.5)
    self.assertRaises(ValueError, Pagel_lambda_MLE, np.array([[3],[2],[1]]), np.array([[8, 6],[6, 8],[4, 5]]), 0.5)
    self.assertRaises(ValueError, Pagel_lambda_MLE, np.array([[3],[2]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 0.5)
    self.assertRaises(ValueError, Pagel_lambda_MLE, np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), -1)
    self.assertRaises(ValueError, Pagel_lambda_MLE, np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 2)
  
  def test_Pagel_lambda_MLE_types(self): # TypeError
    self.assertRaises(TypeError, Pagel_lambda_MLE, [[3],[2],[1]], np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 0.5)
    self.assertRaises(TypeError, Pagel_lambda_MLE, np.array([['a'],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 0.5)
    self.assertRaises(TypeError, Pagel_lambda_MLE, np.array([[3],[2],[1]]), [[8, 6, 4],[6, 8, 5],[4, 5, 6]], 0.5)
    self.assertRaises(TypeError, Pagel_lambda_MLE, np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 'a']]), 0.5)
    self.assertRaises(TypeError, Pagel_lambda_MLE, np.array([[3],[2],[1]]), np.array([[8, 6, 4],[6, 8, 5],[4, 5, 6]]), 'a')
  
  def test_Covariance(self):
    # def Covariance(bac_tree)
    self.assertAlmostEqual(np.array_equal(Covariance(Tree("(A:1,(B:1,(C:1,D:1):0.5):0.5);")), np.array([[1, 0, 0, 0],[0, 1.5, 0.5, 0.5],[0, 0.5, 2, 1],[0, 0.5, 1, 2]])), 1)

  def test_Covariance_types(self): # TypeError
    self.assertRaises(TypeError, Covariance, 'a')
  
  def test_Found_Pagel_Maximumlikelihood(self):
    # def Found_Pagel_Maximumlikelihood(X, tree, stepSize, startSearch = 0, EndSearch = 1)
    self.assertAlmostEqual(np.array_equal(Seedplant_sanity_assistance(), np.array([[-206.66,0.49],[58.99, 0.76],[-521.91, 0.91],[-83.07, 0.95],[-18.07, 0.67]])), 1)
  
  def test_Found_Pagel_Maximumlikelihood_values(self): # ValueError
    self.assertRaises(ValueError, Found_Pagel_Maximumlikelihood, np.array([[3,2],[2,1],[1,0]]), Tree("(B:1,(C:1,D:1):0.5);"), 0.1, 0, 1)
    self.assertRaises(ValueError, Found_Pagel_Maximumlikelihood, np.array([[3],[2],[1]]), Tree("(A:1,(B:1,(C:1,D:1):0.5):0.5);"), 0.1, 0, 1)
    self.assertRaises(ValueError, Found_Pagel_Maximumlikelihood, np.array([[3],[2],[1]]),Tree("(B:1,(C:1,D:1):0.5);"), -0.1, 0, 1)
    self.assertRaises(ValueError, Found_Pagel_Maximumlikelihood, np.array([[3],[2],[1]]),Tree("(B:1,(C:1,D:1):0.5);"), 0.1, -1, 1)
    self.assertRaises(ValueError, Found_Pagel_Maximumlikelihood, np.array([[3],[2],[1]]),Tree("(B:1,(C:1,D:1):0.5);"), 0.1, 2, 1)
    self.assertRaises(ValueError, Found_Pagel_Maximumlikelihood, np.array([[3],[2],[1]]),Tree("(B:1,(C:1,D:1):0.5);"), 0.1, 0, -1)
    self.assertRaises(ValueError, Found_Pagel_Maximumlikelihood, np.array([[3],[2],[1]]),Tree("(B:1,(C:1,D:1):0.5);"), 0.1, 0, 2)
    self.assertRaises(ValueError, Found_Pagel_Maximumlikelihood, np.array([[3],[2],[1]]),Tree("(B:1,(C:1,D:1):0.5);"), 0.1, 1, 0)

  def test_Found_Pagel_Maximumlikelihood_types(self): # TypeError
    self.assertRaises(TypeError, Found_Pagel_Maximumlikelihood, np.array([[3],[2],['a']]), Tree("(B:1,(C:1,D:1):0.5);"), 0.1, 0, 1)
    self.assertRaises(TypeError, Found_Pagel_Maximumlikelihood, 'a', Tree("(B:1,(C:1,D:1):0.5);"), 0.1, 0, 1)
    self.assertRaises(TypeError, Found_Pagel_Maximumlikelihood, np.array([[3],[2],[1]]), 'a', 0.1, 0, 1)
    self.assertRaises(TypeError, Found_Pagel_Maximumlikelihood, np.array([[3],[2],[1]]), Tree("(B:1,(C:1,D:1):0.5);"), 'a', 0, 1)
    self.assertRaises(TypeError, Found_Pagel_Maximumlikelihood, np.array([[3],[2],[1]]), Tree("(B:1,(C:1,D:1):0.5);"), 0.1, 'a', 1)
    self.assertRaises(TypeError, Found_Pagel_Maximumlikelihood, np.array([[3],[2],[1]]), Tree("(B:1,(C:1,D:1):0.5);"), 0.1, 0, 'a')
# Unittest Finished