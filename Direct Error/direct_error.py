# September 11, 2022

# The purpose of this is to compute how similar two songs or soundwaves are by reading a wav file of each then using direct error to determine their MSEs

# Importing PyTorch
import torch
import torchaudio
import torchaudio.functional
import torchaudio.transforms
# Importing the optimizations
from numba import jit, prange
# Importing the helper functions
from pitch_modules import complex_MSE, analysis, static
# Flag to use the optimizations on the function
@jit(nogil=True, parallel=True, fastmath=True)
def pitch_detection_comparison(id1, id2):
  # Loading the songs, with fs being the sample frequency
  # The sample frequency is unused, as it is unnecessary for the method of direct error
  p1, fs1 = torchaudio.load(id1)
  p2, fs2 = torchaudio.load(id2)
  # Reading how many channels each song has
  channel_count1 = len(p1)
  channel_count2 = len(p2)
  out = []
  # Iterating over the channels
  # If one song has more channels than the other, then we only take the MSE of the song with more channels, when we get to a channel count that exists only in one song
  for i in prange(max(channel_count1, channel_count2)):
    if i <= channel_count1 and i <= channel_count2:
        out.append(analysis(p1[i], p2[i]))
    elif i <= channel_count1 and i > channel_count2:
      out.append(static(p1[i]))
    elif i > channel_count1 and i <= channel_count2:
      out.append(static(p2[i]))
    else:
      break
  # Finally, we determine the MSE of the MSEs we've gotten for each channel
  # The closer the ans is to 0, the more similar the two songs are
  ans = complex_MSE(out)
  # Due to PyTorch's implementation of the load function, ans is a tensor quantity. As such, we must use .item() to return the tensor in numeric form
  return ans.item()
# Test Cases
# song1.wav was onclassical_demo_demicheli_geminiani_pieces_allegro-in-f-major_small-version.wav from the Fast Fourier Transform Folder
# song2.wav was elysium_the-young-false-man_small-version_live-and_restored.wav from the Fast Fourier Transform Folder
print(pitch_detection_comparison('song1.wav', 'song2.wav'))
print(pitch_detection_comparison('song1.wav', 'song1.wav'))
print(pitch_detection_comparison('song2.wav', 'song1.wav'))
print(pitch_detection_comparison('song2.wav', 'song2.wav'))
