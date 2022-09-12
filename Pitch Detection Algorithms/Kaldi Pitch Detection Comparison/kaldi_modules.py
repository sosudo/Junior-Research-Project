# September 11, 2022

# This acts as a helper to kaldi_pitch_detection_comparison.py, featuring the MSE functions and appending functions used by it

# Importing the optimzations
from numba import jit, prange
# Calculates the mean squared error for a dataset with complex numbers, finding the magnitude instead of the absolute value
# Flag to use the optimizations
@jit(nogil=True, parallel=True, fastmath=True)
def complex_MSE(data):
  sum = 0
  for i in data:
    # abs() finds both the absolute value and the magnitude depending on the input type
    sum += abs(i)**2
  return sum/len(data)
# Does the analysis on two songs
# Flag to use the optimizations
@jit(nogil=True, parallel=True, fastmath=True)
def analysis(one, two):
  # Finding the length of each data strand
  l1 = len(one)
  l2 = len(two)
  out = []
  # If their sizes aren't equal, we can append them individually
  for i in prange(max(l1, l2)):
    if i < l1 and i < l2:
      out.append(two[i]-one[i])
    elif i < l1 and i > l2:
      out.append(one[i])
    elif i > l1 and i < l2:
      out.append(two[i])
    else:
      break
  # Return the MSE of the error list
  return complex_MSE(out)
# Does the analysis on one song
# Flag to use the optimizations
@jit(nogil=True, parallel=True, fastmath=True)
def static(song):
  # Since the error of just one stand and 0 will be 0, we can immediately calculate the MSE of the error by simply finding the MSE of the strand
  return complex_MSE(song)
