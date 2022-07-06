from scipy.io import wavfile
from stft_modules import complex_MSE, analysis, static
def short_time_fourier_transform_comparison(id1, id2):
  fs1, song1 = wavfile.read(id1)
  fs2, song2 = wavfile.read(id2)
  channel_count1 = song1.shape[1]
  channel_count2 = song2.shape[1]
  out = []
  for i in range(max(channel_count1, channel_count2)):
    if i <= channel_count1 and i <= channel_count2:
      out.append(analysis(fs1, fs2, song1.T[i], song2.T[i]))
    elif i <= channel_count1 and i > channel_count2:
      out.append(static(fs1, song1.T[i]))
    elif i > channel_count1 and i <= channel_count2:
      out.append(static(fs2, song2.T[i]))
    else:
      break
  ans = complex_MSE(out)
  return ans
print(short_term_fourier_transform_comparison('song1.wav', 'song1.wav'))
