# Importing the wavfile module to read songs
from scipy.io import wavfile
# Importing the helper functions
from fourier_modules import complex_MSE, analysis, static
def fast_fourier_transform_comparison(id1, id2):
  # Reading the songs, fs is the sampling frequency, which we don't need in this case
  fs1, song1 = wavfile.read(id1)
  fs2, song2 = wavfile.read(id2)
  # Reading how many channels each song has
  channel_count1 = song1.shape[1]
  channel_count2 = song2.shape[1]
  out = []
  # Iterating over the channels
  # If one song has more channels than the other, then we only take the MSE of the song with more channels, when we get to a channel count that exists only in one song
  for i in range(max(channel_count1, channel_count2)):
    if i <= channel_count1 and i <= channel_count2:
      out.append(analysis(song1.T[i], song2.T[i]))
    elif i <= channel_count1 and i > channel_count2:
      out.append(static(song1.T[i]))
    elif i > channel_count1 and i <= channel_count2:
      out.append(static(song2.T[i]))
    else:
      break
  # Finally, we determine the MSE of the MSEs we've gotten for each channel
  # The closer the ans is to 0, the more similar the two songs are
  ans = complex_MSE(out)
  return ans
print(fast_fourier_transform_comparison('song1.wav', 'song1.wav'))
