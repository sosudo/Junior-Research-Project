# July 6, 2022

# This acts as a helper to short_time_fourier_transform_comparison.py, featuring the MSE functions and appending functions used by it

# Import the signal module, which includes the STFT function we need to make the comparison
from scipy import signal
# Calculates the mean squared error for a dataset with complex numbers, finding the magnitude instead of the absolute value
def complex_MSE(data):
  sum = 0
  for i in data:
    # abs() find both the absolute value and the magnitude depending on the input type
    sum += abs(i)**2
  return sum/len(data)
# Acts as a helper function to the analysis function, since resizing the inner ndarrays is prevented
def analysis_helper(Zxx1, Zxx2):
  # Copying the STFTs for each song so that they're editable
  stft1 = Zxx1.copy()
  stft2 = Zxx2.copy()
  # If one STFT's size isn't the same as the other, we can resize them to match each other
  if len(stft1) != len(stft2):
    if len(stft1) > len(stft2):
      stft2.resize(len(stft1))
    else:
      stft1.resize(len(stft2))
  # Finding the error list
  diff = stft2-stft1
  # Return the MSE of the error list
  return complex_MSE(diff)
# Does the analysis on two songs
def analysis(fs1, fs2, song1, song2):
  # Creating the STFTs of each song
  f1, t1, Zxx1 = signal.stft(song1, fs1)
  f2, t2, Zxx2 = signal.stft(song2, fs2)
  diff = []
  # Iterating through the STFTs as long as they are within the correct length
  for i in range(max(len(Zxx1), len(Zxx2))):
    if i <= len(Zxx1) and i <= len(Zxx2):
      diff.append(analysis_helper(Zxx1[i], Zxx2[i]))
    elif i <= len(Zxx1) and i > len(Zxx2):
      diff.append(complex_MSE(Zxx1[i]))
    elif i > len(Zxx2) and i <= len(Zxx1):
      diff.append(complex_MSE(Zxx2[i]))
    else:
      break
  # Returning the MSE of the MSEs found by iterating through the time-windows of the STFTs
  return complex_MSE(diff)
# Does the analysis on one song
def static(fs, song):
  # Create the STFT
  f, t, Zxx = signal.stft(song, fs)
  # Since the error of just one STFT and 0 will be 0, we can immediately calculate the MSE of the error by simply finding the MSE of the STFT
  return complex_MSE(Zxx)
