# June 28, 2022

# This acts as a helper to fast_fourier_transform_comparison.py, featuring the MSE functions and appending functions used by it

# Import the fft module we'll need to compare our waves
from scipy.fftpack import fft
# Calculates the mean squared average for a dataset with complex numbers, finding the magnitude instead of the absolute value
def complex_MSE(data):
  sum = 0
  for i in data:
    # abs() find both the absolute value and the magnitude depending on the input type
    sum += abs(i)**2
  return sum/len(data)
# Does the analysis on two songs
def analysis(song1, song2):
  # Creating the FFTs of each song
  f1 = fft(song1)
  f2 = fft(song2)
  # If one FFT's size isn't the same as the other, we can resize them to match each other
  if len(f1) != len(f2):
    if len(f1) > len(f2):
      f2.resize(len(f1))
    else:
      f1.resize(len(f2))
  # Finding the error list
  diff = f2-f1
  # Return the MSE of the error list
  return complex_MSE(diff)
# Does the analysis on one song
def static(song):
  # Create the FFT
  f = fft(song)
  # Since the error of just one FFT and 0 will be 0, we can immediately calculate the MSE of the error by simply finding the MSE of the FFT
  return complex_MSE(f)
