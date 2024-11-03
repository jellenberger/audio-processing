import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

# Load the audio file
input_file = "media/input1.wav"
output_file = "media/output_audio_filtered.wav"
y, sr = librosa.load(input_file, sr=None)


# Define a Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Apply bandpass filter to isolate voice frequencies (e.g., 300-3000 Hz for speech)
lowcut = 300.0
highcut = 3000.0
filtered_audio = bandpass_filter(y, lowcut, highcut, sr)


# Perform spectral gating (simple spectral subtraction)
def reduce_noise(y, sr, n_std_thresh=1.0):
    stft = librosa.stft(y)
    magnitude, phase = np.abs(stft), np.angle(stft)

    # Estimate noise threshold from quiet sections (assumes noise is low-energy)
    noise_thresh = np.mean(magnitude, axis=1) + n_std_thresh * np.std(magnitude, axis=1)

    # Mask frequencies below the noise threshold
    mask = magnitude > noise_thresh[:, None]
    clean_magnitude = mask * magnitude

    # Reconstruct the signal
    clean_stft = clean_magnitude * np.exp(1j * phase)
    return librosa.istft(clean_stft)



# Apply noise reduction
cleaned_audio = reduce_noise(filtered_audio, sr)

# Normalize the audio to ensure it's audible
cleaned_audio = cleaned_audio / np.max(np.abs(cleaned_audio))  # normalize to -1 to 1

# Save the output
sf.write(output_file, cleaned_audio, sr)
print(f"Filtered and cleaned audio saved as {output_file}")
