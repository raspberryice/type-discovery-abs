'''
B-cube function from: 
https://github.com/m-wiesner/BCUBED/blob/master/B3score/b3.py
'''

# ========================== B3 ===============================================
# This function determines the extrinsic clustering quality b-cubed measure 
# using a set of known labels for a data set, and cluster assignmnets of 
# each data point stored as either vector or cell arrays where each
# cell/row represets a data point in which an array containing class or cluster
# assignments is stored. The multi-class variant should not yet be used and 
# does not scale well.
#
# Inputs
# ------------------
#       L: An NxM matrix containing class labels for each data point. Each 
#          row represents the ith data point and each column contains a 0, or 1
#          in the jth column indicating membership of label j.
#          Alternatively, if hard class labels are available, L can be input 
#          as an Nx1 vector where each entry is its class label
#
#       K: Defined identically to L except for this variable stores cluster
#          assignments for each data point
#
# Outputs
#--------------------------------
# f_measure: This F-measure using the b-cubed metric
# precision: The b-cubed precision
# recall: The b-cubed recall
#
#-----------------------------------------------
# Author: Matthew Wiesner
# Email : wiesner@jhu.edu 
# Institute: Johns Hopkins University Electrical and Computer Engineering
#
# Refences: DESCRIPTION OF THE UPENN CAMP SYSTEM AS USED FOR COREFERENCE,
#           Breck Baldwin, Tom Morton, Amit Bagga, Jason Baldridge, 
#           Raman Chandraseker, Alexis Dimitriadis, Kieran Snyder, 
#           Magdalena Wolska, Institute for Research in Cognitive Science
#
#           A.A. Aroch-Villarruel Pattern Recognition 6th Mexican Conference,
#           MCPR 2014 Proceedings Paper, p.115
# ----------------------------------------------
import sys 
import numpy as np

# Calculate precision and recall for each class     
def compute_class_precision_recall(L,K):
  '''
    Compute the partitions matrix P which stores the size
    of the intersection of elements belong to Label i and Cluster j
    in the (i,j)-th entry of P

    Input:
      L -- Numpy array of Labels or numpy 2d array with shape (1,N_L) or (N_L,1)
      K -- Numpy array of Clusters or numpy 2d array with shape (1,N_K) or (N_K,1)

    Output:
      P -- Numpy ndarray |\L| x |\K| Partitions Matrix, where |\L| is the
          size of the label set, and \K is the number of clusters
  ''' 
  # Make everything nicely formatted. Ignore skipped labels or clusters  
  _,L = np.unique(np.array(L),return_inverse=True)
  _,K = np.unique(np.array(K),return_inverse=True)

  # Check that there are the same number of labels and clusters 
  if(len(L) != len(K)):
    sys.stderr.write("Labels and clusters are not of the same length.")
    sys.exit(1)

  # Extract some useful variables that will make things easier to read.
  # 1. Number of total elements to cluster
  # 2. Number of distinct labels
  # 3. Number of distinct clusters
  num_elements = len(L)
  num_labels   = L.max() + 1
  num_clusters = K.max() + 1

  # Create binary num_elements x num_labels / num_clusters assignment matrices. 
  X_L = np.tile(L, (num_labels,1) ).T
  X_K = np.tile(K, (num_clusters,1) ).T

  L_j = np.equal( np.tile(np.arange(num_labels),(num_elements,1))   , X_L ).astype(float)
  K_j = np.equal( np.tile(np.arange(num_clusters),(num_elements,1)) , X_K ).astype(float)   

  # Create the partitions matrix which has an element for the 
  # intersection of label i, and cluster j. The element of the matrix is the
  # Number of elements in that partition.
  P_ij = np.dot(L_j.T,K_j) 

  # Summing over the appropriate axes gives the total number of elements
  # in each class label (S_i) or cluster T_i
  S_i  = P_ij.sum(axis=1)
  T_i  = P_ij.sum(axis=0)

  # Calculate Class recall and precision
  R_i  = ( P_ij * P_ij ).sum(axis=1) / ( S_i * S_i )
  P_i  = ( P_ij.T * P_ij.T ).sum(axis=1) / ( T_i * T_i )

  return [(P_i , R_i) , (S_i , T_i)]

# Calculate b3 metrics  
def calc_b3(L , K , class_norm=False, beta=1.0):
  '''
    Implements the BCUBED algorithm according to the DESCRIPTION OF THE UPENN
    CAMP SYSTEM AS USED FOR COREFERENCE, Breck Baldwin, Tom Morton, 
    Amit Bagga, Jason Baldridge, Raman Chandraseker, Alexis Dimitriadis, 
    Kieran Snyder, Magdalena Wolska, Institute for Research in Cognitive 
    Science.

    Usage:
      from B3 import B3
      import numpy as np

      score = B3()
      L = np.array([1,3,3,3,3,4,2,2,2,3,3])
      K = np.array([1,2,3,4,5,5,5,6,2,1,1])

      # Standard BCUBED (Weight each element equally)
      [fmeasure, precision, recall] = score.calc_b3(L,K)
      
      # Equivalence class normalization (Weight each class equally)
      [fmeasure, precision, recall] = score.calc_b3(L,K,class_norm=True)

      # Different weighting schemes for fmeasure
      [fmeasure, precision, recall] = score.calc_b3(L,K,beta=2.0)
      [fmeasure, precision, recall] = score.calc_b3(L,K,beta=0.5)


    Computes the precision, recall, and fmeasure from the Class level
    precision, and recall arrays. Two types are possible. One weights all
    classes equally while the other weights each element equally.

    Input:
      L -- Numpy array of Labels or numpy 2d array with shape (1,N_L) or (N_L,1)
      K -- Numpy array of Clusters or numpy 2d array with shape (1,N_K) or (N_K,1)

      options:
        class_norm: Decides whether to weight the precision by class or by entity
        beta: Harmonic mean weighting
  '''
  
  # Compute per equivalence class precision and recall
  precision_recall , class_sizes = compute_class_precision_recall(L,K)
  
  # Two methods of obtaining overall precision and recall
  if(class_norm == True):
    precision = precision_recall[0].sum() / class_sizes[1].size 
    recall    = precision_recall[1].sum() / class_sizes[0].size
  else:
    precision = ( precision_recall[0] * class_sizes[1] ).sum() / class_sizes[1].sum()
    recall    = ( precision_recall[1] * class_sizes[0] ).sum() / class_sizes[0].sum()
  
  # f_measure with option beta to weight the precision and recall asymmetrically.
  f_measure = (1 + beta**2) * (precision * recall) /( (beta**2) * precision + recall ) 
  
  return [f_measure,precision,recall]      





