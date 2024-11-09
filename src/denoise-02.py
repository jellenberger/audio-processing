import numpy as np
from scipy import signal
import soundfile as sf

def enhance_ssb_voice(input_file, output_file):
    """
    Enhance SSB voice audio with minimal noise reduction to preserve voice naturality.
    
    Args:
        input_file (str): Path to input WAV file
        output_file (str): Path to output WAV file
    """
    # Read the audio file using soundfile instead of wavfile
    audio_data, sample_rate = sf.read(input_file)
    
    # Convert to float32 if not already
    audio_data = audio_data.astype(np.float32)
    
    # Normalize input (only if necessary)
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data = audio_data / max_val
    
    # Bandpass filter for voice frequencies (300-3300 Hz for SSB)
    nyquist = sample_rate / 2
    low_cut = 300 / nyquist  # Changed back to 300 Hz
    high_cut = 3300 / nyquist  # Kept at 3300 Hz
    b, a = signal.butter(2, [low_cut, high_cut], btype='band')
    
    # Apply bandpass filter
    filtered_audio = signal.filtfilt(b, a, audio_data)
    
    # Gentler noise reduction
    frame_length = int(0.05 * sample_rate)  # 50ms frames
    hop_length = frame_length // 4
    
    # Function to estimate noise floor
    def estimate_noise(signal_data, num_frames=5):
        # Ensure we don't try to read beyond the signal length
        max_frames = min(num_frames, len(signal_data) // frame_length)
        frames = np.array_split(signal_data[:max_frames * frame_length], max_frames)
        return np.mean([np.abs(np.fft.rfft(frame)) for frame in frames], axis=0)
    
    # Estimate noise profile
    noise_profile = estimate_noise(filtered_audio)
    
    # Process audio in overlapping frames
    enhanced_audio = np.zeros_like(filtered_audio)
    window = signal.windows.hann(frame_length)
    
    # Ensure we don't process beyond the actual signal length
    for i in range(0, len(filtered_audio) - frame_length, hop_length):
        frame = filtered_audio[i:i + frame_length]
        windowed_frame = frame * window
        
        # FFT
        spectrum = np.fft.rfft(windowed_frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Adjusted spectral subtraction
        oversubtraction = 0.5
        # Add reduced noise floor to prevent complete silence (reduced from 0.1 to 0.05)
        magnitude = np.maximum(magnitude - oversubtraction * noise_profile[:len(magnitude)], 
                             0.05 * noise_profile[:len(magnitude)])  # Reduced noise floor
        
        # Reconstruct signal
        enhanced_frame = np.fft.irfft(magnitude * np.exp(1j * phase))
        enhanced_audio[i:i + frame_length] += enhanced_frame * window
    
    # Normalize
    enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio))
    
    # Very gentle compression
    threshold = 0.5
    ratio = 1.5
    makeup_gain = 0.8
    
    enhanced_audio = np.where(
        np.abs(enhanced_audio) > threshold,
        np.sign(enhanced_audio) * (threshold + (np.abs(enhanced_audio) - threshold) / ratio),
        enhanced_audio
    )
    
    # Apply reduced makeup gain and ensure we don't clip
    enhanced_audio = np.clip(enhanced_audio * makeup_gain * 0.8, -0.8, 0.8)
    
    # Write output file using soundfile
    sf.write(output_file, enhanced_audio, sample_rate, subtype='FLOAT')

if __name__ == "__main__":
    enhance_ssb_voice("media/input.wav", "media/output.wav")