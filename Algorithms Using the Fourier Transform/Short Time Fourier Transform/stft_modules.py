from scipy import signal
import numpy as np
def complex_MSE(data):
  sum = 0
  for i in data:
    sum += abs(i)**2
  return sum/len(data)
def analysis_helper(Zxx1, Zxx2):
  stft1 = Zxx1.copy()
  stft2 = Zxx2.copy()
  if len(stft1) != len(stft2):
    if len(stft1) > len(stft2):
      stft2.resize(len(stft1))
    else:
      stft1.resize(len(stft2))
  diff = stft2-stft1
  return complex_MSE(diff)
def analysis(fs1, fs2, song1, song2):
  f1, t1, Zxx1 = signal.stft(song1, fs1)
  f2, t2, Zxx2 = signal.stft(song2, fs2)
  diff = []
  for i in range(max(len(Zxx1), len(Zxx2))):
    if i <= len(Zxx1) and i <= len(Zxx2):
      diff.append(analysis_helper(Zxx1[i], Zxx2[i]))
    elif i <= len(Zxx1) and i > len(Zxx2):
      diff.append(complex_MSE(Zxx1[i]))
    elif i > len(Zxx2) and i <= len(Zxx1):
      diff.append(complex_MSE(Zxx2[i]))
    else:
      break
  return complex_MSE(diff)
def static(fs, song):
  f, t, Zxx = signal.stft(song, fs)
  return complex_MSE(Zxx)
