
import numpy as np
from scipy.io import wavfile


# Get the recorded signals.
freq1, data1 = wavfile.read('wav/mic1.wav')
freq2, data2 = wavfile.read('wav/mic2.wav')

# Assemble into a 2D array. Each row is a recording from a mic.
x = np.vstack((data1, data2))

# Just to test.
y = x

# Make new wav files containing the unmixed signals.
wavfile.write('wav/unmix1.wav', freq1, y[0].astype(np.int16))
wavfile.write('wav/unmix2.wav', freq1, y[1].astype(np.int16))
