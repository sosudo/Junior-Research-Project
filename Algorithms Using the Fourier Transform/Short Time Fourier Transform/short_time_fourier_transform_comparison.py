# July 6, 2022

# The purpose of this is to compute how similar two songs or soundwaves are by reading a wav file of each then using the STFT to determine their MSEs

# Importing the wavfile module to read songs
from scipy.io import wavfile
# Importing the optimzations
from numba import jit, prange
# Importing the helper functions
from stft_modules import complex_MSE, analysis, static
# Flag to use the optimizations
@jit(nogil=True, parallel=True)
def short_time_fourier_transform_comparison(id1, id2):
  # Reading the songs, fs is the sampling frequency
  fs1, song1 = wavfile.read(id1)
  fs2, song2 = wavfile.read(id2)
  # Reading how many channels each song has
  channel_count1 = song1.shape[1]
  channel_count2 = song2.shape[1]
  out = []
  # Iterating over the channels
  # If one song has more channels than the other, then we only take the MSE of the song with more channels, when we get to a channel count that exists only in one song
  for i in prange(max(channel_count1, channel_count2)):
    if i <= channel_count1 and i <= channel_count2:
      out.append(analysis(fs1, fs2, song1.T[i], song2.T[i]))
    elif i <= channel_count1 and i > channel_count2:
      out.append(static(fs1, song1.T[i]))
    elif i > channel_count1 and i <= channel_count2:
      out.append(static(fs2, song2.T[i]))
    else:
      break
  # Finally, we determine the MSE of the MSEs we've gotten for each channel
  # The closer the ans is to 0, the more similar the two songs are
  ans = complex_MSE(out)
  return ans
# Test Cases
# This uses the same songs as the FFT algorithm, but the wavfiles have been renamed
print(short_time_fourier_transform_comparison('song1.wav', 'song1.wav'))
print(short_time_fourier_transform_comparison('song1.wav', 'song2.wav'))
print(short_time_fourier_transform_comparison('song2.wav', 'song1.wav'))
print(short_time_fourier_transform_comparison('song2.wav', 'song2.wav'))
